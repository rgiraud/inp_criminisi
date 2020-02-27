//
//PM with SP 2D
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

#ifndef MAX
#define MAX(a, b) ((a)>(b)?(a):(b))
#endif

#ifndef MIN
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

#define EPSILON 0.0000001

#ifndef enc_type
#define enc_type unsigned char
#endif




void compute_mask_size(unsigned char * mask, int s, int * mask_size) {
    
    int count = 0;
    for (int i=0; i<s; i++)
        count += (int) mask[i];
    
    *mask_size = count;
    
}


void sobel_operator_uchar(int y, int x, int h, unsigned char * mask, float * grad_y, float * grad_x){
    
    float gx = (float) ((float) mask[(y+1)+x*h] - (float) mask[(y-1)+x*h]);
    float gy = (float) ((float) mask[y+(x+1)*h] - (float) mask[y+(x-1)*h]);
    
    *grad_y = -gy;
    *grad_x = -gx;
    
}


void sobel_operator_float(int y, int x, int h, int w, unsigned char * mask, float * img, float * grad_y, float * grad_x, int d_yn, int d_xn){
    
    float gx = 0;
    float gy = 0;
    
    if ((d_yn == 0) && (d_xn == 0)) {
        gy = img[y+1+x*h] - img[y-1+x*h];
        gx = img[y+(x+1)*h] - img[y+(x-1)*h];
    }
    else {
        if ((y+d_yn-1 < 0) || (mask[y+d_yn-1 + (x+d_xn)*h] == 1)) {
            if ( (y+d_yn+1<h) && (mask[y+d_yn+1+ (x+d_xn)*h] == 0) ) {
                gy = img[y+d_yn+1+(x+d_xn)*h] - img[y+d_yn+(x+d_xn)*h];
            }
            else
                gy = 0;
        }
        else
            gy = img[y+d_yn+(x+d_xn)*h] - img[y+d_yn-1+(x+d_xn)*h];
        
        if ((x+d_xn-1 < 0) || (mask[y+d_yn + (x+d_xn-1)*h] == 1)) {
            if ( (x+d_xn+1<w) && (mask[y+d_yn+ (x+d_xn+1)*h] == 0) ) {
                gx = img[y+d_yn+(x+d_xn+1)*h] - img[y+d_yn+(x+d_xn)*h];
            }
            else
                gx = 0;
        }
        else
            gx = img[y+d_yn+(x+d_xn)*h] - img[y+d_yn+(x+d_xn-1)*h];
        
    }
    
    *grad_y = gy;
    *grad_x = gx;
    
}


void sobel_gradient_mask(unsigned char * mask, int h, int w, unsigned char * mask_contour, float *grad_map){
    
    for (int y=1; y<h-1; y++) {
        for (int x=1; x<w-1; x++) {
            
            if (mask_contour[y+x*h]){
                
                float grad_x = 0;
                float grad_y = 0;
                
                sobel_operator_uchar(y,x,h,mask,&grad_y,&grad_x);
                
                grad_map[y+x*h] = grad_y;
                grad_map[y+x*h+h*w] = grad_x;
                
            }
        }
    }
    
    
}

void sobel_gradient_img(float * img, int h, int w, unsigned char * mask, float *grad_map, float *grad_contour){
    
    for (int y=1; y<h-1; y++) {
        for (int x=1; x<w-1; x++) {
            
            int d_ny = (int) grad_contour[y+x*h];
            int d_nx = (int) grad_contour[y+x*h+h*w];
            
            if ((y+d_ny>=0) && (y+d_ny<h) && (x+d_nx>=0) && (x+d_nx<w)) {
                
                float grad_x = 0;
                float grad_y = 0;
                
                sobel_operator_float(y,x,h,w,mask,img,&grad_y,&grad_x,d_ny,d_nx);
                
                grad_map[y+x*h] = grad_y;
                grad_map[y+x*h+h*w] = grad_x;
                
            }
        }
    }
    
}



void priority_queue_update(float * img, int h, int w, unsigned char * mask, unsigned char * mask_contour,
        int *y_ptr, int * x_ptr, float *grad_img, float *grad_contour, float * priority_map, int patch_bw){
    
    float max_grad = -2;
    
    float pb2 = (float) (2*patch_bw+1)*(2*patch_bw+1);
    float alpha = 1;
    float grad_x = 0;
    float grad_y = 0;
    
    for (int dy=-patch_bw-1; dy<=patch_bw+1; dy++) {
        for (int dx=-patch_bw-1; dx<=patch_bw+1; dx++) {
            int y = *y_ptr+dy;
            int x = *x_ptr+dx;
            if ((y>=0) && (y<h) && (x>=0) && (x<w)) {
                if (mask[y+x*h]) {
                    int is_contour = 0;
                    for (int yy=-1; yy<=1; yy++) {
                        for (int xx=-1; xx<=1; xx++) {
                            if ((y+yy>=0) && (y+yy<h) && (x+xx>=0) && (x+xx<w)) {
                                if (mask[y+yy+(x+xx)*h]==0) {
                                    yy = 2;
                                    xx = 2;
                                    is_contour = 1;
                                }
                            }
                            
                        }
                    }
                    mask_contour[y+x*h] = is_contour;
                }
            }
        }
    }
    
    //recompute grad contour & grad_img
    for (int dy=-patch_bw-1; dy<=patch_bw+1; dy++) {
        for (int dx=-patch_bw-1; dx<=patch_bw+1; dx++) {
            int y = *y_ptr+dy;
            int x = *x_ptr+dx;
            if ( (y>=1) && (y<h-1) && (x>=1) && (x<w-1)) {
                if (mask_contour[y+x*h]) {
                    //update grad_contour
                    float dx = 0;
                    float dy = 0;
                    sobel_operator_uchar(y,x,h,mask,&dy,&dx);
                    grad_contour[y+x*h] = dy;
                    grad_contour[y+x*h+h*w] = dx;
                    //update contour img
                    grad_x = 0;
                    grad_y = 0;
                    sobel_operator_float(y,x,h,w,mask,img,&grad_y,&grad_x,(int) dy,(int) dx);
                    grad_img[y+x*h] = grad_y;
                    grad_img[y+x*h+h*w] = grad_x;
                }
            }
        }
    }
    
    
    for (int dy=-patch_bw-1; dy<=patch_bw+1; dy++) {
        for (int dx=-patch_bw-1; dx<=patch_bw+1; dx++) {
            
            int y = *y_ptr+dy;
            int x = *x_ptr+dx;
            
            if ( (y>=0) && (y<h) && (x>=0) && (x<w)) {
                
                float C = 0;
                float D = 0;
                if (mask_contour[y+x*h]){
                    
                    //Confidence
                    for (int ddy=-patch_bw; ddy<=patch_bw; ddy++) {
                        for (int ddx=-patch_bw; ddx<=patch_bw; ddx++) {
                            
                            if ( (y+ddy>=0) && (y+ddy<h) && (x+ddx>=0) && (x+ddx<w)) {
                                if ((mask[y+ddy+(x+ddx)*h] == 0)) {
                                    C += 1;
                                }
                            }
                        }
                    }
                    
                    //Data term
                    //vector norm
                    float ny = grad_contour[y+x*h];
                    float nx = grad_contour[y+x*h+h*w];
                    float norm_xy = (sqrt(ny*ny+nx*nx)+0.01);
                    ny = ny/norm_xy;
                    nx = nx/norm_xy;
                    
                    D = fabsf(ny*grad_img[y+x*h+h*w] - nx*grad_img[y+x*h])    +  0.00001;
                    
                    C = C/pb2;
                    D = D/alpha;
                    priority_map[y+x*h] = C*D;
                    
                }
            }
            
        }
    }
    
    //find max
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            if (mask_contour[y+x*h]){
                if (priority_map[y+x*h] >= max_grad) {
                    max_grad = priority_map[y+x*h];
                    *y_ptr = y;
                    *x_ptr = x;
                }
            }
        }
    }
    
}


void priority_queue(int h, int w, unsigned char * mask, unsigned char * mask_contour,
        int *y_ptr, int * x_ptr, float *grad_img, float *grad_contour, float * priority_map, int patch_w){
    
    
    float max_grad = -2;
    
    float pb2 = (float) (2*patch_w+1)*(2*patch_w+1);
    float alpha = 1;
    
    for (int y=1; y<h-1; y++) {
        for (int x=1; x<w-1; x++) {
            
            float C = 0;
            float D = 0;
            if (mask_contour[y+x*h]){
                
                //Confidence
                for (int dy=-patch_w; dy<=patch_w; dy++) {
                    for (int dx=-patch_w; dx<=patch_w; dx++) {
                        
                        if ( (y+dy>=0) && (y+dy<h) && (x+dx>=0) && (x+dx<w)) {
                            
                            if ((mask[y+dy+(x+dx)*h] == 0)) {
                                C += 1;
                            }
                            
                        }
                        
                    }
                }
                C = C/pb2;
                
                //Data term
                float ny = grad_contour[y+x*h];
                float nx = grad_contour[y+x*h+h*w];
                float norm_xy = (sqrt(ny*ny+nx*nx)+0.01);
                ny = ny/norm_xy;
                nx = nx/norm_xy;
                
                D = fabsf(ny*grad_img[y+x*h+h*w] - nx*grad_img[y+x*h])    +  0.00001;
                
                D = D*1/alpha;
                
                
                priority_map[y+x*h] = C*D;
                
                
            }
            
        }
    }
    
    //find max
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            if (mask_contour[y+x*h]){
                if (priority_map[y+x*h] > max_grad) {
                    max_grad = priority_map[y+x*h];
                    *y_ptr = y;
                    *x_ptr = x;
                }
            }
        }
    }
    
}




void criminisi_inpainting(float *a, unsigned char * mask, unsigned char * mask_contour, float * a_g,
        int ha, int wa, int da, int patch_w, int patch_bw, int s, int * nnf_inp, int * map_queue,
        float * a_i, unsigned char * mask_i){
    
    
    int y = 0;
    int x = 0;
    int mask_size = 0;
    int count = 0;
    float * grad_img = (float *) calloc(ha*wa*2,sizeof(float));
    float * grad_contour = (float *) calloc(ha*wa*2,sizeof(float));
    float * priority_map = (float *) calloc(ha*wa,sizeof(float));
    
    
    //Compute size of the mask to fill
    compute_mask_size(mask, ha*wa, &mask_size);
    
    if (mask_size > 0){
        sobel_gradient_mask(mask, ha, wa, mask_contour, grad_contour);
        sobel_gradient_img(a_g, ha, wa, mask, grad_img, grad_contour);
        priority_queue(ha, wa, mask, mask_contour, &y, &x, grad_img, grad_contour, priority_map, patch_w);
    }
    
    printf("pixels to inpaint ... %d ...", mask_size);
    mexEvalString("drawnow");
    
    
    while (mask_size > 0) {
        
        if (mask_size % 100 == 0) {
            printf("%d ...", mask_size);
            mexEvalString("drawnow");
        }
        
        
        //process selected patch (y,x)
        int best_match_pos = 0;
        float best_dist = FLT_MAX;
        
        
        //patch search
        int min_yb = MAX(y-s,patch_w);
        int min_xb = MAX(x-s,patch_w);
        int max_yb = MIN(y+s,ha-patch_w);
        int max_xb = MIN(x+s,wa-patch_w);
        
        
        
        for (int yb=min_yb; yb<max_yb; yb++) {
            for (int xb=min_xb; xb<max_xb; xb++) {
                
                float dist = 0;
                
                
                for (int dx=-patch_w; dx<=patch_w; dx++) {
                    for (int dy=-patch_w; dy<=patch_w; dy++) {
                        
                        if ( (yb+dy>=0) && (yb+dy<ha) && (xb+dx>=0) &&  (xb+dx<wa)) {
                            
                            if (mask_i[yb+dy + (xb+dx)*ha]==1)
                                dist += 9999999;
                            
                            if ( (y+dy>=0) && (y+dy<ha) && (x+dx>=0) &&  (x+dx<wa)) {
                                if (mask[y+dy+(x+dx)*ha]==0) {
                                    
                                    for (int dd=0; dd<da; dd++) {
                                        dist += (a[y+dy+(x+dx)*ha+ha*wa*dd] - a_i[yb+dy+(xb+dx)*ha+ha*wa*dd])*(a[y+dy+(x+dx)*ha+ha*wa*dd] - a_i[yb+dy+(xb+dx)*ha+ha*wa*dd]);
                                    }
                                }
                            }
                        }
                        if (dist > best_dist){
                            dx = patch_w +1;
                            dy = patch_w +1;
                        }
                        
                    }
                }
                
                if (dist < best_dist) {
                    best_dist = dist;
                    best_match_pos = yb+xb*ha;
                }
                
            }
        }
        //best match pos
        int ybb = (best_match_pos%(ha*wa))%ha;
        int xbb = (best_match_pos%(ha*wa))/ha;
        
        
        
        //copy of best match (patch_bw)
        for (int dx=-patch_bw; dx<=patch_bw; dx++) {
            for (int dy=-patch_bw; dy<=patch_bw; dy++) {
                if ( (y+dy>=0) && (y+dy<ha) && (x+dx>=0) &&  (x+dx<wa)) {
                    if ( (ybb+dy>=0) && (ybb+dy<ha) && (xbb+dx>=0) &&  (xbb+dx<wa)) {
                        if (mask[y+dy + (x+dx)*ha] == 1) {
                            
                            a[y+dy + (x+dx)*ha] = a_i[best_match_pos+dy+dx*ha];
                            a[y+dy + (x+dx)*ha + ha*wa] = a_i[best_match_pos+dy+dx*ha+ha*wa];
                            a[y+dy + (x+dx)*ha + ha*wa*2] = a_i[best_match_pos+dy+dx*ha+ha*wa*2];
                            
                            mask[y+dy + (x+dx)*ha] = 0;
                            mask_contour[y+dy+(x+dx)*ha] = 0;
                            
                            //output nnf
                            nnf_inp[y+dy+(x+dx)*ha] = ybb;
                            nnf_inp[y+dy+(x+dx)*ha+ha*wa] = xbb;
                            
                            map_queue[y+dy+(x+dx)*ha] = count;
                            
                        }
                    }
                }
            }
        }
        
        
        //Loop stuff
        int tmp = mask_size;
        compute_mask_size(mask,ha*wa,&mask_size);
        if (tmp == mask_size)
            mask_size = 0;
        
        count += 1;
        
        priority_queue_update(a, ha, wa, mask, mask_contour, &y, &x, grad_img, grad_contour, priority_map, patch_bw);
        
    }
    
    
    printf("\n");
    mexEvalString("drawnow");
    
    free(priority_map);
    free(grad_img);
    free(grad_contour);
}



//////////////////////////////////////////////////////////////////////////
/////////////////////////////////// MAIN /////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void mexFunction(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[]){
    
    
    
    
    //Inputs
    float* a = (float*) mxGetPr(prhs[0]);
    unsigned char* mask = (unsigned char*) mxGetPr(prhs[1]);
    unsigned char* mask_contour = (unsigned char*) mxGetPr(prhs[2]);
    
    int idx = 3;
    int patch_w = (int) mxGetScalar(prhs[idx++]);
    int patch_bw = (int) mxGetScalar(prhs[idx++]);
    int s = (int) mxGetScalar(prhs[idx++]);
    
    
    const int *a_dims = mxGetDimensions(prhs[0]);
    int ha = a_dims[0];
    int wa = a_dims[1];
    int da = a_dims[2];
    
    //a*mask (to make sure there is inpainting)
    for (int i=0; i<ha*wa; i++) {
        if (mask[i]) {
            for (int d=0; d<da; d++) {
                a[i+d*ha*wa] = 0;
            }
        }
    }
    
    //Initial mask and image copy
    float * a_i = (float *) calloc(ha*wa*da,sizeof(float));
    unsigned char * mask_i = (unsigned char *) calloc(ha*wa,sizeof(unsigned char));
    for (int i=0; i<ha*wa; i++) {
        for (int d=0; d<da; d++) {
            a_i[i+d*ha*wa] = a[i+d*ha*wa];
        }
        mask_i[i] = mask[i];
    }
    
    //level of gray image (for gradient computation)
    float * a_g = (float *) calloc(ha*wa,sizeof(float));
    
    if (da == 3) {
        for (int i=0; i<ha; i++) {
            for (int j=0; j<wa; j++) {
                a_g[i+j*ha] = a[i+j*ha] + a[i+j*ha+ha*wa] +
                        a[i+j*ha+ha*wa*2];
                a_g[i+j*ha] /= 3;
            }
        }
    }
    
    //OUTPUT
    int dims[3];
    dims[0] = ha;
    dims[1] = wa;
    dims[2] = 3;
    
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    float* img_out = (float*)mxGetPr(plhs[0]);
    
    
    plhs[1] = mxCreateNumericArray(3, dims, mxINT32_CLASS, mxREAL);
    int* nnf_inp = (int*)mxGetPr(plhs[1]);
    //Complementary outputs
    for (int i=0; i<ha; i++) {
        for (int j=0; j<wa; j++) {
            if (mask[i+j*ha]==0) {
                nnf_inp[i+j*ha] = i;
                nnf_inp[i+j*ha+ha*wa] = j;
                nnf_inp[i+j*ha+2*ha*wa] = 0;
            }
        }
    }
    
    
    
    //map prioriry queue
    dims[0] = ha;
    dims[1] = wa;
    plhs[2] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    int* map_queue = (int*)mxGetPr(plhs[2]);
    
    
    // INPAINTING
    criminisi_inpainting(a, mask, mask_contour, a_g, ha, wa, da, patch_w, patch_bw, s, nnf_inp, map_queue, a_i, mask_i);
    
    
    //copy_output
    for (int i=0; i<ha*wa*3; i++)
        img_out[i] = a[i];
    
    
    
    //free
    free(a_g);
    free(a_i);
    free(mask_i);
    
}