/*!
 *  Copyright (c) 2015 by Contributors
 * \file im2rec.cc
 * \brief convert images into image recordio format
 *  Image Record Format: zeropad[64bit] imid[64bit] img-binary-content
 *  The 64bit zero pad was reserved for future purposes
 *
 *  Image List Format: unique-image-index label[s] path-to-image
 * \sa dmlc/recordio.h
 */
#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
#include <opencv2/opencv.hpp>
#include "../src/io/image_recordio.h"
#include <random>

double n = 0, mean = 0, M2 = 0, stdev = 0;
double nc = 0, meanc[3] = {0, 0, 0}, M2c[3] = {0, 0, 0}, stdevc[3] = {0, 0, 0};

void OnlineMeanDeviation(cv::Mat res)
{
  for (int i = 0; i < res.cols; i++)
  {
    for (int j = 0; j < res.rows; j++)
    {

    	nc++;
      cv::Vec3b intensity = res.at<cv::Vec3b>(j, i);
      for (int k = 0; k < res.channels(); k++)
      {
        double x = intensity.val[k];

        n = n + 1;
        double delta = x - mean;
        mean = mean + delta / (double)n;
        M2 = M2 + delta * (x - mean);

        double deltac = x - meanc[k];
        meanc[k] += deltac / (double)nc;
        M2c[k] += deltac * (x - meanc[k]);
      }

    }
  }

  stdev = sqrt(M2 / (n - 1));

      for (int k = 0; k < res.channels(); k++)
      {
	      stdevc[k] = sqrt(M2c[k] / (nc - 1));      	
      }
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width, int inter_method)
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        int scaledHeight = floor(height * scale);
        scaledHeight = scaledHeight > 0 ? scaledHeight : 1;
        roi.height = scaledHeight;
        roi.y = floor(( target_width - roi.height ) / 2);
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        int scaledWidth = floor(width * scale);
        scaledWidth = scaledWidth > 0 ? scaledWidth : 1;
        roi.width = scaledWidth;
        roi.x = floor(( target_width - roi.width ) / 2);
    }

    cv::resize( img, square( roi ), roi.size(), 0, 0, inter_method);

    return square;
}


/*!
 *\brief get interpolation method with given inter_method, 0-CV_INTER_NN 1-CV_INTER_LINEAR 2-CV_INTER_CUBIC
 *\ 3-CV_INTER_AREA 4-CV_INTER_LANCZOS4 9-AUTO(cubic for enlarge, area for shrink, bilinear for others) 10-RAND(0-4)
 */
int GetInterMethod(int inter_method, int old_width, int old_height, int new_width, int new_height, std::mt19937& prnd) {
    if (inter_method == 9) {
        if (new_width > old_width && new_height > old_height) {
            return 2;  // CV_INTER_CUBIC for enlarge
        } else if (new_width <old_width && new_height < old_height) {
            return 3;  // CV_INTER_AREA for shrink
        } else {
            return 1;  // CV_INTER_LINEAR for others
        }
    } else if (inter_method == 10) {
        std::uniform_int_distribution<size_t> rand_uniform_int(0, 4);
        return rand_uniform_int(prnd);
    } else {
        return inter_method;
    }
}
int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: <image.lst> <image_root_dir> <output.rec> [additional parameters in form key=value]\n"\
           "Possible additional parameters:\n"\
           "\tcolor=USE_COLOR[default=1] Force color (1), gray image (0) or keep source unchanged (-1).\n"\
           "\tresize=newsize resize the shorter edge of image to the newsize, original images will be packed by default\n"\
           "\tlabel_width=WIDTH[default=1] specify the label_width in the list, by default set to 1\n"\
           "\tnsplit=NSPLIT[default=1] used for part generation, logically split the image.list to NSPLIT parts by position\n"\
           "\tpart=PART[default=0] used for part generation, pack the images from the specific part in image.list\n"\
           "\tcenter_crop=CENTER_CROP[default=0] specify whether to crop the center image to make it square.\n"\
           "\tquality=QUALITY[default=100] JPEG quality for encoding (1-100, default: 80) or PNG compression for encoding (1-9, default: 3).\n"\
           "\tencoding=ENCODING[default='.jpg'] Encoding type. Can be '.jpg' or '.png'\n"\
           "\tinter_method=INTER_METHOD[default=9] NN(0) BILINEAR(1) CUBIC(2) AREA(3) LANCZOS4(4) AUTO(9) RAND(10).\n"\
           "\tunchanged=UNCHANGED[default=0] Keep the original image encoding, size and color. If set to 1, it will ignore the others parameters.\n");
    return 0;
  }
  int label_width = 1;
  int new_size = -1;
  int nsplit = 1;
  int partid = 0;
  int center_crop = 0;
  int quality = 100;
  int color_mode = CV_LOAD_IMAGE_COLOR;
  int unchanged = 0;
  int inter_method = 9; // CV_INTER_CUBIC;
  std::string encoding(".jpg");
  for (int i = 4; i < argc; ++i) {
    char key[128], val[128];
    if (sscanf(argv[i], "%[^=]=%s", key, val) == 2) {
      if (!strcmp(key, "resize")) new_size = atoi(val);
      if (!strcmp(key, "label_width")) label_width = atoi(val);
      if (!strcmp(key, "nsplit")) nsplit = atoi(val);
      if (!strcmp(key, "part")) partid = atoi(val);
      if (!strcmp(key, "center_crop")) center_crop = atoi(val);
      if (!strcmp(key, "quality")) quality = atoi(val);
      if (!strcmp(key, "color")) color_mode = atoi(val);
      if (!strcmp(key, "encoding")) encoding = std::string(val);
      if (!strcmp(key, "unchanged")) unchanged = atoi(val);
      if (!strcmp(key, "inter_method")) inter_method = atoi(val);
    }
  }
  // Check parameters ranges
  if (color_mode != -1 && color_mode != 0 && color_mode != 1) {
      LOG(FATAL) << "Color mode must be -1, 0 or 1.";
  }
  if (encoding != std::string(".jpg") && encoding != std::string(".png")) {
      LOG(FATAL) << "Encoding mode must be .jpg or .png.";
  }
  if (new_size > 0) {
    LOG(INFO) << "New Image Size: Short Edge " << new_size;
  } else {
    LOG(INFO) << "Keep origin image size";
  }
  if (center_crop) {
    LOG(INFO) << "Center cropping to square";
  }
  if (color_mode == 0) {
    LOG(INFO) << "Use gray images";
  }
  if (color_mode == -1) {
    LOG(INFO) << "Keep original color mode";
  }
  LOG(INFO) << "Encoding is " << encoding;

  if (encoding == std::string(".png") and quality > 9) {
      quality = 3;
  }
  if (inter_method != 1) {
      switch (inter_method) {
        case 0:
            LOG(INFO) << "Use inter_method CV_INTER_NN";
            break;
        case 2:
            LOG(INFO) << "Use inter_method CV_INTER_CUBIC";
            break;
        case 3:
            LOG(INFO) << "Use inter_method CV_INTER_AREA";
            break;
        case 4:
            LOG(INFO) << "Use inter_method CV_INTER_LANCZOS4";
            break;
        case 9:
            LOG(INFO) << "Use inter_method mod auto(cubic for enlarge, area for shrink)";
            break;
        case 10:
            LOG(INFO) << "Use inter_method mod rand(nn/bilinear/cubic/area/lanczos4)";
           break;
        default:
            LOG(INFO) << "Unkown inter_method";
            return 0;
      }
  }
  std::random_device rd;
  std::mt19937 prnd(rd());
  using namespace dmlc;
  const static size_t kBufferSize = 1 << 20UL;
  std::string root = argv[2];
  mxnet::io::ImageRecordIO rec;
  double tstart = dmlc::GetTime();
  dmlc::InputSplit *flist = dmlc::InputSplit::
      Create(argv[1], partid, nsplit, "text");
  std::ostringstream os;
  if (nsplit == 1) {
    os << argv[3];
  } else {
    os << argv[3] << ".part" << std::setw(3) << std::setfill('0') << partid;
  }
  LOG(INFO) << "Write to output: " << os.str();
  dmlc::Stream *fo = dmlc::Stream::Create(os.str().c_str(), "w");
  LOG(INFO) << "Output: " << os.str();
  dmlc::RecordIOWriter writer(fo);
  std::string fname, path, blob;
  std::vector<unsigned char> decode_buf;
  std::vector<unsigned char> encode_buf;
  std::vector<int> encode_params;
  if (encoding == std::string(".png")) {
      encode_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
      encode_params.push_back(quality);
      LOG(INFO) << "PNG encoding compression: " << quality;
  } else {
      encode_params.push_back(CV_IMWRITE_JPEG_QUALITY);
      encode_params.push_back(quality);
      LOG(INFO) << "JPEG encoding quality: " << quality;
  }
  dmlc::InputSplit::Blob line;
  
	size_t imcnt = 0;
  while (flist->NextRecord(&line)) {
    std::string sline(static_cast<char*>(line.dptr), line.size);
    std::istringstream is(sline);
    if (!(is >> rec.header.image_id[0] >> rec.header.label)) continue;
    for (int k = 1; k < label_width; ++k) {
      float tmp;
      CHECK(is >> tmp)
          << "Invalid ImageList, did you provide the correct label_width?";
    }
    CHECK(std::getline(is, fname));
    // eliminate invalid chars in the end
    while (fname.length() != 0 &&
           (isspace(*fname.rbegin()) || !isprint(*fname.rbegin()))) {
      fname.resize(fname.length() - 1);
    }
    // eliminate invalid chars in beginning.
    const char *p = fname.c_str();
    while (isspace(*p)) ++p;
    path = root + p;
    // use "r" is equal to rb in dmlc::Stream
    dmlc::Stream *fi = dmlc::Stream::Create(path.c_str(), "r");
    rec.SaveHeader(&blob);
    decode_buf.clear();
    size_t imsize = 0;
    while (true) {
      decode_buf.resize(imsize + kBufferSize);
      size_t nread = fi->Read(BeginPtr(decode_buf) + imsize, kBufferSize);
      imsize += nread;
      decode_buf.resize(imsize);
      if (nread != kBufferSize) break;
    }
    delete fi;
    if (unchanged != 1) {
      cv::Mat img = cv::imdecode(decode_buf, color_mode);
      CHECK(img.data != NULL) << "OpenCV decode fail:" << path;
      cv::Mat res = img;

/*
      if (new_size > 0) {
        if (center_crop) {
          if (img.rows > img.cols) {
            int margin = (img.rows - img.cols)/2;
            img = img(cv::Range(margin, margin+img.cols), cv::Range(0, img.cols));
          } else {
            int margin = (img.cols - img.rows)/2;
            img = img(cv::Range(0, img.rows), cv::Range(margin, margin + img.rows));
          }
        }
        int interpolation_method = 1;
        if (img.rows > img.cols) {
            if (img.cols != new_size) {
                interpolation_method = GetInterMethod(inter_method, img.cols, img.rows, new_size, img.rows * new_size / img.cols, prnd);
                cv::resize(img, res, cv::Size(new_size, img.rows * new_size / img.cols), 0, 0, interpolation_method);
            } else {
                res = img.clone();
            }
        } else {
            if (img.rows != new_size) {
                interpolation_method = GetInterMethod(inter_method, img.cols, img.rows, new_size * img.cols / img.rows, new_size, prnd);
                cv::resize(img, res, cv::Size(new_size * img.cols / img.rows, new_size), 0, 0, interpolation_method);
            } else {
                res = img.clone();
            }
        }
      }
*/
      int interpolation_method = GetInterMethod(inter_method, img.cols, img.rows, new_size, new_size, prnd);      
      res = GetSquareImage(img, new_size, interpolation_method);
      OnlineMeanDeviation(res);
//      cv::imwrite("pic.jpg", res);
//      getline();

      encode_buf.clear();
      CHECK(cv::imencode(encoding, res, encode_buf, encode_params));
      size_t bsize = blob.size();
      blob.resize(bsize + encode_buf.size());
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(encode_buf), encode_buf.size());
    } else {
      size_t bsize = blob.size();
      blob.resize(bsize + decode_buf.size());
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(decode_buf), decode_buf.size());
    }
    writer.WriteRecord(BeginPtr(blob), blob.size());
    // write header
    ++imcnt;
    if (imcnt % 5000 == 0) {
      LOG(INFO) << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
	  LOG(INFO) << "Mean: " << mean << " , Standard Deviation: " << stdev;
	  LOG(INFO) << "Blue Mean: " << meanc[0] << " , Standard Deviation: " << stdevc[0];
	  LOG(INFO) << "Green Mean: " << meanc[1] << " , Standard Deviation: " << stdevc[1];
	  LOG(INFO) << "Red Mean: " << meanc[2] << " , Standard Deviation: " << stdevc[2];
    }
  }

  LOG(INFO) << "Total: " << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
  LOG(INFO) << "Mean: " << mean << " , Standard Deviation: " << stdev;
  LOG(INFO) << "Blue Mean: " << meanc[0] << " , Standard Deviation: " << stdevc[0];
  LOG(INFO) << "Green Mean: " << meanc[1] << " , Standard Deviation: " << stdevc[1];
  LOG(INFO) << "Red Mean: " << meanc[2] << " , Standard Deviation: " << stdevc[2];

  delete fo;
  delete flist;
  return 0;
}
