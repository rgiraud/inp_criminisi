
clear all
close all
clc

%Compilation
mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./inpainting_criminisi.c -outdir ./

%Inputs
img = double(imread('img/olive.png'))/255;
mask = uint8(imread('img/olive_mask.png'));              
[h,w,z] = size(img);


%Display inputs
figure,
axes_p(1) = subplot(221);
imagesc(min(1,img+repmat(double(mask),[1 1 3])))
title('Image to inpaint')
axes_p(2) = subplot(222);
imagesc((mask))
title('Mask')
drawnow;

% Parameters
pw = 5;    %search for similar patch size (2*pw+1)
pbw = 3;   %copy patch of size 2*pbw+1, pbw <= pw
s = 50;    %search area of size 2*s+1)

%Get initial mask_contour
mask_contour = zeros(h,w);
for i=1:h
    for j=1:w
        if (mask(i,j) == 1)
            win = mask(max(1,i-1):min(h,i+1),max(1,j-1):min(w,j+1));
            if (mean(win(:)) < 1)
                mask_contour(i,j) = 1;
            end
        end
    end
end

%Inpainting
[imgout, nnf_inp, map_queue] = inpainting_criminisi(single(img), uint8(mask), uint8(mask_contour), pw, pbw, s);

%display result
axes_p(3) = subplot(223);
imagesc(imgout)
title('Inpainted image')
axes_p(4) = subplot(224);
imagesc(map_queue)
title('Map queue')
linkaxes(axes_p)

