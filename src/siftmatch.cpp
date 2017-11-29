#include "siftmatch.h"

//SIFT算法头文件
//加extern "C"，告诉编译器按C语言的方式编译和连接
extern "C"
{
#include "imgfeatures.h"
#include "kdtree.h"
#include "minpq.h"
#include "sift.h"
#include "utils.h"
#include "xform.h"
}

//在k-d树上进行BBF搜索的最大次数
/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

//目标点与最近邻和次近邻的距离的比值的阈值，若大于此阈值，则剔除此匹配点对
//通常此值取0.6，值越小找到的匹配点对越精确，但匹配数目越少
/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

//窗口名字符串
#define IMG1 "img1"
#define IMG2 "img2"
#define IMG1_FEAT "img1 feature key"
#define IMG2_FEAT "img2 feature key"
#define IMG_MATCH1 "距离比值筛选后的匹配结果"
#define IMG_MATCH2 "RANSAC筛选后的匹配结果"
#define IMG_MOSAIC_TEMP "临时拼接图像"
#define IMG_MOSAIC_SIMPLE "简易拼接图"
#define IMG_MOSAIC_BEFORE_FUSION "重叠区域融合前"
#define IMG_MOSAIC_PROC "处理后的拼接图"


SiftMatch::SiftMatch()
{    
    open_image_number = 0;//打开图片的个数

    //设定横竖排的单选按钮的勾选状态，默认是横排

    img1 = nullptr;
    img2 = nullptr;
    img1_Feat = nullptr;
    img2_Feat = nullptr;
    stacked = nullptr;
    stacked_ransac = nullptr;
    H = nullptr;
    xformed = nullptr;

    verticalStackFlag = false;//显示匹配结果的合成图像默认是横向排列
}

SiftMatch::~SiftMatch()
{
    restart();
}

//打开图片
void SiftMatch::input_img(const char* img1, const char* img2)
{
    this->img1 = cvLoadImage(img1);
    this->img2 = cvLoadImage(img2);

}
void SiftMatch::input_img(IplImage *img1, IplImage *img2)
{
    this->img1 = img1;
    this->img2 = img2;
}

//特征点检测
void SiftMatch::feature_detect()
{
    img1_Feat = cvCloneImage(img1);//复制图1，深拷贝，用来画特征点
    img2_Feat = cvCloneImage(img2);//复制图2，深拷贝，用来画特征点

    //默认提取的是LOWE格式的SIFT特征点
    //提取并显示第1幅图片上的特征点
    n1 = sift_features( img1, &feat1 );//检测图1中的SIFT特征点,n1是图1的特征点个数
//    export_features("feature1.txt",feat1,n1);//将特征向量数据写入到文件
//    draw_features( img1_Feat, feat1, n1 );//画出特征点
//    cvNamedWindow(IMG1_FEAT);//创建窗口
//    cvShowImage(IMG1_FEAT,img1_Feat);//显示

    //提取并显示第2幅图片上的特征点
    n2 = sift_features( img2, &feat2 );//检测图2中的SIFT特征点，n2是图2的特征点个数
//    export_features("feature2.txt",feat2,n2);//将特征向量数据写入到文件
//    draw_features( img2_Feat, feat2, n2 );//画出特征点
//    cvNamedWindow(IMG2_FEAT);//创建窗口
//    cvShowImage(IMG2_FEAT,img2_Feat);//显示
}

//特征匹配
void SiftMatch::feature_match()
{
    stacked = stack_imgs_horizontal(img1, img2);//合成图像，显示经距离比值法筛选后的匹配结果

    //根据图1的特征点集feat1建立k-d树，返回k-d树根给kd_root
    kd_root = kdtree_build( feat1, n1 );

    Point pt1,pt2;//连线的两个端点
    double d0,d1;//feat2中每个特征点到最近邻和次近邻的距离
    int matchNum = 0;//经距离比值法筛选后的匹配点对的个数

    //遍历特征点集feat2，针对feat2中每个特征点feat，选取符合距离比值条件的匹配点，放到feat的fwd_match域中
    for(int i = 0; i < n2; i++ )
    {
        feat = feat2+i;//第i个特征点的指针
        //在kd_root中搜索目标点feat的2个最近邻点，存放在nbrs中，返回实际找到的近邻点个数
        int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if( k == 2 )
        {
            d0 = descr_dist_sq( feat, nbrs[0] );//feat与最近邻点的距离的平方
            d1 = descr_dist_sq( feat, nbrs[1] );//feat与次近邻点的距离的平方
            //若d0和d1的比值小于阈值NN_SQ_DIST_RATIO_THR，则接受此匹配，否则剔除
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
            {   //将目标点feat和最近邻点作为匹配点对
                pt2 = Point( cvRound( feat->x ), cvRound( feat->y ) );//图2中点的坐标
                pt1 = Point( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//图1中点的坐标(feat的最近邻点)
                if(verticalStackFlag)//垂直排列
                    pt2.y += img1->height;//由于两幅图是上下排列的，pt2的纵坐标加上图1的高度，作为连线的终点
                else
                    pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
                cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//画出连线
                matchNum++;//统计匹配点对的个数
                feat2[i].fwd_match = nbrs[0];//使点feat的fwd_match域指向其对应的匹配点
            }
        }
        free( nbrs );//释放近邻数组
    }

    //显示并保存经距离比值法筛选后的匹配图
//    cvNamedWindow(IMG_MATCH1);//创建窗口
//    cvShowImage(IMG_MATCH1,stacked);//显示
    //保存匹配图

    //利用RANSAC算法筛选匹配点,计算变换矩阵H，
    //无论img1和img2的左右顺序，H永远是将feat2中的特征点变换为其匹配点，即将img2中的点变换为img1中的对应点
    H = ransac_xform(feat2,n2,FEATURE_FWD_MATCH,lsq_homog,4,0.01,homog_xfer_err,3.0,&inliers,&n_inliers);

    //若能成功计算出变换矩阵，即两幅图中有共同区域
    if( H )
    {
//        std::cout<<"经RANSAC算法筛选后的匹配点对个数："<<n_inliers<<std::endl;

        if(verticalStackFlag)//将2幅图片合成1幅图片,img1在上，img2在下
            stacked_ransac = stack_imgs( img1, img2 );//合成图像，显示经RANSAC算法筛选后的匹配结果
        else//将2幅图片合成1幅图片,img1在左，img2在右
            stacked_ransac = stack_imgs_horizontal(img1, img2);//合成图像，显示经RANSAC算法筛选后的匹配结果

        //img1LeftBound = inliers[0]->fwd_match->x;//图1中匹配点外接矩形的左边界
        //img1RightBound = img1LeftBound;//图1中匹配点外接矩形的右边界
        //img2LeftBound = inliers[0]->x;//图2中匹配点外接矩形的左边界
        //img2RightBound = img2LeftBound;//图2中匹配点外接矩形的右边界

        int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图

        //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线
        for(int i=0; i<n_inliers; i++)
        {
            feat = inliers[i];//第i个特征点
            pt2 = Point(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标

            /*找匹配点区域的边界
            if(pt1.x < img1LeftBound) img1LeftBound = pt1.x;
            if(pt1.x > img1RightBound) img1RightBound = pt1.x;
            if(pt2.x < img2LeftBound) img2LeftBound = pt2.x;
            if(pt2.x > img2RightBound) img2RightBound = pt2.x;//*/

            //统计匹配点的左右位置关系，来判断图1和图2的左右位置关系
            if(pt2.x > pt1.x)
                invertNum++;

            if(verticalStackFlag)//垂直排列
                pt2.y += img1->height;//由于两幅图是上下排列的，pt2的纵坐标加上图1的高度，作为连线的终点
            else//水平排列
                pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
            cvLine(stacked_ransac,pt1,pt2,CV_RGB(255,0,255),1,8,0);//在匹配图上画出连线
        }

        //绘制图1中包围匹配点的矩形
        //cvRectangle(stacked_ransac,cvPoint(img1LeftBound,0),cvPoint(img1RightBound,img1->height),CV_RGB(0,255,0),2);
        //绘制图2中包围匹配点的矩形
        //cvRectangle(stacked_ransac,cvPoint(img1->width+img2LeftBound,0),cvPoint(img1->width+img2RightBound,img2->height),CV_RGB(0,0,255),2);

//        cvNamedWindow(IMG_MATCH2);//创建窗口
//        cvShowImage(IMG_MATCH2,stacked_ransac);//显示经RANSAC算法筛选后的匹配图
        //保存匹配图

        /*程序中计算出的变换矩阵H用来将img2中的点变换为img1中的点，正常情况下img1应该是左图，img2应该是右图。
          此时img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x < pt1.x
          若用户打开的img1是右图，img2是左图，则img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x > pt1.x
          所以通过统计对应点变换前后x坐标大小关系，可以知道img1是不是右图。
          如果img1是右图，将img1中的匹配点经H的逆阵H_IVT变换后可得到img2中的匹配点*/

        //若pt2.x > pt1.x的点的个数大于内点个数的80%，则认定img1中是右图
        if(invertNum > n_inliers * 0.8)
        {
//            std::cout<<"img1中是右图"<<std::endl;
            CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//变换矩阵的逆矩阵
            //求H的逆阵H_IVT时，若成功求出，返回非零值
            if( cvInvert(H,H_IVT) )
            {
                cvReleaseMat(&H);//释放变换矩阵H，因为用不到了
                H = cvCloneMat(H_IVT);//将H的逆阵H_IVT中的数据拷贝到H中
                cvReleaseMat(&H_IVT);//释放逆阵H_IVT
                //将img1和img2对调
                IplImage * temp = img2;
                img2 = img1;
                img1 = temp;
                //cvShowImage(IMG1,img1);
                //cvShowImage(IMG2,img2);
            }
            else//H不可逆时，返回0
            {
                cvReleaseMat(&H_IVT);//释放逆阵H_IVT
            }
        }
    }
    else //无法计算出变换矩阵，即两幅图中没有重合区域
    {
        std::cout<<"WARNING:"<<"There are no public areas in the two plots"<<std::endl;
    }
}

//计算图2的四个角经矩阵H变换后的坐标
void SiftMatch::CalcFourCorner()
{
    //计算图2的四个角经矩阵H变换后的坐标
    double v2[]={0,0,1};//左上角
    double v1[3];//变换后的坐标值
    CvMat V2 = cvMat(3,1,CV_64FC1,v2);
    CvMat V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);//矩阵乘法
    leftTop.x = cvRound(v1[0]/v1[2]);
    leftTop.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

    //将v2中数据设为左下角坐标
    v2[0] = 0;
    v2[1] = img2->height;
    V2 = cvMat(3,1,CV_64FC1,v2);
    V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);
    leftBottom.x = cvRound(v1[0]/v1[2]);
    leftBottom.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

    //将v2中数据设为右上角坐标
    v2[0] = img2->width;
    v2[1] = 0;
    V2 = cvMat(3,1,CV_64FC1,v2);
    V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);
    rightTop.x = cvRound(v1[0]/v1[2]);
    rightTop.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

    //将v2中数据设为右下角坐标
    v2[0] = img2->width;
    v2[1] = img2->height;
    V2 = cvMat(3,1,CV_64FC1,v2);
    V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);
    rightBottom.x = cvRound(v1[0]/v1[2]);
    rightBottom.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,rightBottom,7,CV_RGB(255,0,0),2);

}

//全景拼接
void SiftMatch::mosaic()
{
    //若能成功计算出变换矩阵，即两幅图中有共同区域，才可以进行全景拼接
    if(H)
    {
        //拼接图像，img1是左图，img2是右图
        CalcFourCorner();//计算图2的四个角经变换后的坐标
        //为拼接结果图xformed分配空间,高度为图1图2高度的较小者，根据图2右上角和右下角变换后的点的位置决定拼接图的宽度
        xformed = cvCreateImage(cvSize(MIN(rightTop.x,rightBottom.x),MIN(img1->height,img2->height)),IPL_DEPTH_8U,3);
        //用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
        cvWarpPerspective(img2,xformed,H,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
//        cvNamedWindow(IMG_MOSAIC_TEMP); //显示临时图,即只将图2变换后的图
//        cvShowImage(IMG_MOSAIC_TEMP,xformed);

        //简易拼接法：直接将将左图img1叠加到xformed的左边
//        xformed_simple = cvCloneImage(xformed);//简易拼接图，可笼子xformed
//        cvSetImageROI(xformed_simple,cvRect(0,0,img1->width,img1->height));
//        cvAddWeighted(img1,1,xformed_simple,0,0,xformed_simple);
//        cvResetImageROI(xformed_simple);
//        cvNamedWindow(IMG_MOSAIC_SIMPLE);//创建窗口
//        cvShowImage(IMG_MOSAIC_SIMPLE,xformed_simple);//显示简易拼接图

        //处理后的拼接图，克隆自xformed
        xformed_proc = cvCloneImage(xformed);

        //重叠区域左边的部分完全取自图1
        cvSetImageROI(img1,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
        cvSetImageROI(xformed,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
        cvSetImageROI(xformed_proc,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
        cvAddWeighted(img1,1,xformed,0,0,xformed_proc);
        cvResetImageROI(img1);
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
//        cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION);
//        cvShowImage(IMG_MOSAIC_BEFORE_FUSION,xformed_proc);//显示融合之前的拼接图

        //采用加权平均的方法融合重叠区域
        int start = MIN(leftTop.x,leftBottom.x) ;//开始位置，即重叠区域的左边界
        double processWidth = img1->width - start;//重叠区域的宽度
        double alpha = 1;//img1中像素的权重
        for(int i=0; i<xformed_proc->height; i++)//遍历行
        {
            const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1中第i行数据的指针
            const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed中第i行数据的指针
            uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc中第i行数据的指针
            for(int j=start; j<img1->width; j++)//遍历重叠区域的列
            {
                //如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据
                if(pixel_xformed[j*3] < 50 && pixel_xformed[j*3+1] < 50 && pixel_xformed[j*3+2] < 50 )
                {
                    alpha = 1;
                }
                else
                {   //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
                    alpha = (processWidth-(j-start)) / processWidth ;
                }
                pixel_xformed_proc[j*3] = pixel_img1[j*3] * alpha + pixel_xformed[j*3] * (1-alpha);//B通道
                pixel_xformed_proc[j*3+1] = pixel_img1[j*3+1] * alpha + pixel_xformed[j*3+1] * (1-alpha);//G通道
                pixel_xformed_proc[j*3+2] = pixel_img1[j*3+2] * alpha + pixel_xformed[j*3+2] * (1-alpha);//R通道
            }
        }
//        cvNamedWindow(IMG_MOSAIC_PROC);//创建窗口
//        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图
//        cvSaveImage("ret0.jpg", xformed_proc);

//        //*重叠区域取两幅图像的平均值，效果不好
//        //设置ROI，是包含重叠区域的矩形
//        cvSetImageROI(xformed_proc,cvRect(MIN(leftTop.x,leftBottom.x),0,img1->width-MIN(leftTop.x,leftBottom.x),xformed_proc->height));
//        cvSetImageROI(img1,cvRect(MIN(leftTop.x,leftBottom.x),0,img1->width-MIN(leftTop.x,leftBottom.x),xformed_proc->height));
//        cvSetImageROI(xformed,cvRect(MIN(leftTop.x,leftBottom.x),0,img1->width-MIN(leftTop.x,leftBottom.x),xformed_proc->height));
//        cvAddWeighted(img1,0.5,xformed,0.5,0,xformed_proc);
//        cvResetImageROI(xformed_proc);
//        cvResetImageROI(img1);
//        cvResetImageROI(xformed); //*/
//        cvSaveImage("ret_camera.jpg", xformed_proc);
//
//        //对拼接缝周围区域进行滤波来消除拼接缝，效果不好
//        //在处理前后的图上分别设置横跨拼接缝的矩形ROI
//        cvSetImageROI(xformed_proc,cvRect(img1->width-10,0,img1->width+10,xformed->height));
//        cvSetImageROI(xformed,cvRect(img1->width-10,0,img1->width+10,xformed->height));
//        cvSmooth(xformed,xformed_proc,CV_MEDIAN,5);//对拼接缝周围区域进行中值滤波
//        cvResetImageROI(xformed);
//        cvResetImageROI(xformed_proc);
//        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图
//        cvSaveImage("ret2.jpg", xformed_proc);
//
//        //想通过锐化解决变换后的图像失真的问题，对于扭曲过大的图像，效果不好
//        double a[]={  0, -1,  0, -1,  5, -1, 0, -1,  0  };//拉普拉斯滤波核的数据
//        CvMat kernel = cvMat(3,3,CV_64FC1,a);//拉普拉斯滤波核
//        cvFilter2D(xformed_proc,xformed_proc,&kernel);//滤波
//        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图
//        cvSaveImage("ret3.jpg", xformed_proc);


        //保存拼接图
//        QString name_xformed = name1;//文件名，原文件名去掉序号后加"_Mosaic"
//        cvSaveImage(name_xformed.replace( name_xformed.lastIndexOf(".",-1)-1 , 1 , "_Mosaic").toAscii().data() , xformed_simple);//保存简易拼接图
//        cvSaveImage(name_xformed.insert( name_xformed.lastIndexOf(".",-1) , "_Proc").toAscii().data() , xformed_proc);//保存处理后的拼接图
//        ui->mosaicButton->setEnabled(false);//禁用全景拼接按钮
    }
}

// 重新选择
void SiftMatch::restart()
{
    //释放并关闭原图1
    if(img1)
    {
        cvReleaseImage(&img1);
        cvDestroyWindow(IMG1);
    }
    //释放并关闭原图2
    if(img2)
    {
        cvReleaseImage(&img2);
        cvDestroyWindow(IMG2);
    }
    //释放特征点图1
    if(img1_Feat)
    {
        cvReleaseImage(&img1_Feat);
        cvDestroyWindow(IMG1_FEAT);
        free(feat1);//释放特征点数组
    }
    //释放特征点图2
    if(img2_Feat)
    {
        cvReleaseImage(&img2_Feat);
        cvDestroyWindow(IMG2_FEAT);
        free(feat2);//释放特征点数组
    }
    //释放距离比值筛选后的匹配图和kd树
    if(stacked)
    {
        cvReleaseImage(&stacked);
        cvDestroyWindow(IMG_MATCH1);
        kdtree_release(kd_root);//释放kd树
    }
    //只有在RANSAC算法成功算出变换矩阵时，才需要进一步释放下面的内存空间
    if( H )
    {
        cvReleaseMat(&H);//释放变换矩阵H
        free(inliers);//释放内点数组

        //释放RANSAC算法筛选后的匹配图
        cvReleaseImage(&stacked_ransac);
        cvDestroyWindow(IMG_MATCH2);

        //释放全景拼接图像
        if(xformed)
        {
            cvReleaseImage(&xformed);
            cvReleaseImage(&xformed_simple);
            cvReleaseImage(&xformed_proc);
            cvDestroyWindow(IMG_MOSAIC_TEMP);
            cvDestroyWindow(IMG_MOSAIC_SIMPLE);
            cvDestroyWindow(IMG_MOSAIC_BEFORE_FUSION);
            cvDestroyWindow(IMG_MOSAIC_PROC);
        }
    }

    open_image_number = 0;//打开图片个数清零
    verticalStackFlag = false;//显示匹配结果的合成图片的排列方向标识复位
}

IplImage* SiftMatch::getRet() {
    return xformed_proc;
}

