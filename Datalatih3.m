 clc; clear; close all; warning off all;

%menetapkan nama folder
nama_folder = 'Data Latih';
%membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
%membaca jumlah file
jumlah_file = numel(nama_file);

%menginisialisasi variabel data_latih
data_latih = zeros(jumlah_file,13);
%melakukan pengolahan citra terhadap seluruh file
for k = 1:jumlah_file
    %membaca file citra rgb
    Img = imread(fullfile(nama_folder,nama_file(k).name));
    % Konversi gambar RGB ke HSV
    gambarAsli = double(Img) / 255.0;
    % Tentukan deviasi standar (sigma) untuk filter Gaussian
    sigma = 1.0; % Anda dapat mengatur nilai sigma sesuai kebutuhan
   % Terapkan filter Gaussian
    filteredImage = imgaussfilt(gambarAsli, sigma);
    % Tentukan nilai gamma (sesuaikan jika diperlukan)
    gamma = 1.5;
    % Terapkan koreksi gamma
    gambarHasil =  filteredImage .^ (1/gamma);
    % Konversi gambar hasil kembali ke format uint8 untuk ditampilkan
    gambarHasil = uint8(gambarHasil * 255);
  
    % Konversi citra ke skala abu-abu jika belum dalam format tersebut
     
    %filter mean
    kernel_size = 5;
    filtered_img = imfilter( gambarHasil,fspecial('average',[kernel_size,kernel_size]));
   

    %Melakukan segmentasi dengan k means clustering
    kmeanseg =  filtered_img;
    %figure, imshow(Img), title('original image');
    % Color-Based Segmentation Using K-Means Clustering
    cform = makecform('srgb2lab');
    lab = applycform(kmeanseg,cform);
    %figure, imshow(lab), title('L*a*b color space');

    ab = double(lab(:,:,2:3));
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,2);
 
    nColors = 3;
    [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
 
    pixel_labels = reshape(cluster_idx,nrows,ncols);
    RGB = label2rgb(pixel_labels);
    %figure, imshow(RGB,[]), title('image labeled by cluster index');

    segmented_images = cell(1,3);
    rgb_label = repmat(pixel_labels,[1 1 3]);
 
for ka = 1:nColors
    color = kmeanseg;
    color(rgb_label ~= ka) = 0;
    segmented_images{ka} = color;
end

%figure,imshow(segmented_images{1}), title('objects in cluster 1');
%figure,imshow(segmented_images{2}), title('objects in cluster 2');
%figure,imshow(segmented_images{3}), title('objects in cluster 3');

% jeruk segmentasi
area_cluster1 = sum(sum(pixel_labels==1));
area_cluster2 = sum(sum(pixel_labels==2));
area_cluster3 = sum(sum(pixel_labels==3));
 
[~,cluster_jeruk] = min([area_cluster1,area_cluster2,area_cluster3]);
jeruk_bw = (pixel_labels==cluster_jeruk);
jeruk_bw = imfill(jeruk_bw,'holes');
jeruk_bw = bwareaopen(jeruk_bw,1000);
 
jeruk = kmeanseg;
R = jeruk(:,:,1);
G = jeruk(:,:,2);
B = jeruk(:,:,3);
R(~jeruk_bw) = 0;
G(~jeruk_bw) = 0;
B(~jeruk_bw) = 0;
jeruk_rgb = cat(3,R,G,B);
%figure, imshow(jeruk_rgb); %title('the jeruk only (RGB Color Space)');
image = jeruk_rgb;
gambar_hsv = rgb2hsv(image);
gray_image = rgb2gray(image);

 % Ambil saluran warna H, S, dan V
Hue = gambar_hsv(:,:,1);
Saturation = gambar_hsv(:,:,2);
Value = gambar_hsv(:,:,3);
HSV = cat(3,Hue,Saturation ,Value);
%figure, imshow(HSV); %title('the jeruk only (RGB Color Space)');


% figure, imshow(HSV); %title('the jeruk only (RGB Color Space)');
 
 image = HSV;
% Ekstraksi fitur warna dari setiap saluran
mean_hue = mean(Hue(:));
mean_saturation = mean(Saturation(:));
mean_value = mean(Value(:));

median_hue = median(Hue(:));
median_saturation = median(Saturation(:));
median_value = median(Value(:));

%mode_hue = mode(Hue);
%mode_saturation = mode(Saturation);
%mode_value = mode(Value);

%variance_hue = var(double(Hue));
%variance_saturation = var(double(Saturation(segmentasi)));
%variance_value = var(double(Value(segmentasi)));

%melakukan ekstraksi ciri tekstur menggunakan metode GLCM

 pixel_dist = 1;
 GLCM = graycomatrix(gray_image,'offset',[0 pixel_dist]);
 stats = graycoprops(GLCM);

 Correlation = mean(stats.Correlation);  
 Energy = mean(stats.Energy);
 Contrast = mean(stats.Contrast);
 Homogeneity = mean(stats.Homogeneity); 

 % Baca gambar biner (misalnya hasil segmentasi)
gambar_biner =  gray_image;
thresimage = imbinarize(gambar_biner, .6);
imcomp = imcomplement(thresimage);
morfo = bwareaopen(imcomp,3000);

% Konversi ke tipe data double
img = double(morfo);

% Operator Robert
robert_mask1 = [1 0; 0 -1];
robert_mask2 = [0 1; -1 0];

% Konvolusi dengan masing-masing kernel
robert_edge1 = conv2(img, robert_mask1, 'same');
robert_edge2 = conv2(img, robert_mask2, 'same');

% Gabungkan kedua hasil
robert_edge = sqrt(robert_edge1.^2 + robert_edge2.^2);

 % Ekstraksi fitur bentuk
    stats = regionprops(robert_edge, 'Area', 'Perimeter', 'Solidity', 'Eccentricity');
    % Ekstraksi fitur bentuk dari setiap objek
for n = 1:numel(stats)
    % Ambil nilai Area dan Perimeter dari objek ke-k
    Area = stats(n).Area;
    %Perimeter = stats(n).Perimeter;
    Solidity = stats.Solidity;
    Eccentricity = stats.Eccentricity;
    % Menghitung fitur bentuk berdasarkan parameter Metric
    %Metric = Perimeter^2 / (4 * pi * Area);
end
    
% Tampilkan hasil ekstraksi fitur bentuk
%disp('Ekstraksi Fitur Bentuk:');
%disp(['Area: ', num2str(area)]);
%disp(['Parameter: ', num2str(parameter)]);
%disp(['Metric: ', num2str(metric)]);
%disp(['Eccentricity: ', num2str(eccentricity)]);

 %menyusun variabel data_uji
 data_latih(k,1) = mean_hue;
 data_latih(k,2) = mean_saturation;
 data_latih(k,3) = mean_value;
 data_latih(k,4) = median_hue;
 data_latih(k,5) = median_saturation;
 data_latih(k,6) = median_value;
  %menyusun variabel data_latih GLCM
 data_latih(k,7) = Correlation;
 data_latih(k,8) = Energy;
 data_latih(k,9) = Contrast;
 data_latih(k,10) = Homogeneity;
 
 %menyusun variabel data latih bentuk
 data_latih(k, 11) = Area;
 %data_latih(k, 12) = Perimeter;
 data_latih(k, 12) = Solidity;
 data_latih(k, 13) = Eccentricity;
 %data_latih(k, 15) = Metric;
 
end

% menetapkan target_latih
    target_latih = cell(jumlah_file,1);
    for k = 1:80
        target_latih{k} = 'Matang A';
    end
    
    for k = 81:160
        target_latih{k} = 'Matang B';
    end
    
    for k = 161:240
        target_latih{k} = 'Matang C';
    end

% melakukan pelatihan menggunakan algoritma MSVM
Mdl = fitcecoc(data_latih, target_latih);

% Melakukan pelatihan menggunakan algoritma MSVM dengan kernel linear
Mdl_linear = fitcecoc(data_latih, target_latih, 'Learners', templateSVM('KernelFunction', 'linear'));

% Melakukan pelatihan menggunakan algoritma MSVM dengan kernel polynomial
Mdl_poly = fitcecoc(data_latih, target_latih, 'Learners', templateSVM('KernelFunction', 'polynomial'));

% Melakukan pelatihan menggunakan algoritma MSVM dengan kernel RBF (Gaussian)
Mdl_rbf = fitcecoc(data_latih, target_latih, 'Learners', templateSVM('KernelFunction', 'rbf'));


% membaca kelas keluaran hasil pelatihan
kelas_keluaran = predict (Mdl,data_latih);

kelas_keluaranlinear = predict (Mdl_linear, data_latih);

kelas_keluaranpoly = predict (Mdl_poly, data_latih);

kelas_keluaranrbf = predict (Mdl_rbf, data_latih);

% menghitung akurasi pelatihan
jumlah_benar = 0;
for k = 1:jumlah_file
    if isequal (kelas_keluaran{k},target_latih{k})
        jumlah_benar = jumlah_benar+1;
    end
end

% menghitung akurasi pelatihan
jumlah_benar1 = 0;
for k = 1:jumlah_file
    if isequal (kelas_keluaranlinear{k},target_latih{k})
        jumlah_benar1 = jumlah_benar1+1;
    end
end

% menghitung akurasi pelatihan
jumlah_benar2 = 0;
for k = 1:jumlah_file
    if isequal (kelas_keluaranpoly{k},target_latih{k})
        jumlah_benar2 = jumlah_benar2+1;
    end
end

% menghitung akurasi pelatihan
jumlah_benar3 = 0;
for k = 1:jumlah_file
    if isequal (kelas_keluaranrbf{k},target_latih{k})
        jumlah_benar3 = jumlah_benar3+1;
    end
end

%Menghitung metrik evaluasi
CM = confusionmat(target_latih, kelas_keluaran);

% Menghitung akurasi
accuracy = sum(diag(CM)) / sum(CM(:));

% Menampilkan hasil
fprintf('Akurasi: %.2f%%\n', accuracy * 100);

akurasi_pelatihan = jumlah_benar/jumlah_file*100;
%menyimpan model naive bayes hasil pelatihan
save Mdl Mdl
save Mdl_linear Mdl_linear 
save Mdl_poly Mdl_poly
save Mdl_rbf Mdl_rbf