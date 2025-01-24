clc; clear; close all; warning off all;

%menetapkan nama folder
nama_folder = 'DataUji73';
%membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
%membaca jumlah file
jumlah_file = numel(nama_file);
%menginisialisasi variabel data_latih
data_uji = zeros(jumlah_file,13);
%melakukan pengolahan citra terhadap seluruh file
for k = 1:jumlah_file
    %membaca file citra rgb
    Img = imread(fullfile(nama_folder,nama_file(k).name));
    % Konversi gambar RGB ke HSV
    gambarAsli = double(Img) / 255.0;
    % Tentukan nilai gamma (sesuaikan jika diperlukan)
    gamma = 1.5;
    % Terapkan koreksi gamma
    gambarHasil = gambarAsli .^ (1/gamma);
    % Konversi gambar hasil kembali ke format uint8 untuk ditampilkan
    gambarHasil = uint8(gambarHasil * 255);
    gambar_hsv = rgb2hsv( gambarHasil);
    % Konversi citra ke skala abu-abu jika belum dalam format tersebut
     
    %filter mean
    kernel_size = 5;
    filtered_img = imfilter( gambarHasil,fspecial('average',[kernel_size,kernel_size]));
    gray_image = rgb2gray(filtered_img);


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

    % Ambil nilai-nilai Area dari semua objek dan gabungkan ke dalam satu array
    
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
 data_uji(k,1) = mean_hue;
 data_uji(k,2) = mean_saturation;
 data_uji(k,3) = mean_value;
 data_uji(k,4) = median_hue;
 data_uji(k,5) = median_saturation;
 data_uji(k,6) = median_value;
  %menyusun variabel data_latih GLCM
 data_uji(k,7) = Correlation;
 data_uji(k,8) = Energy;
 data_uji(k,9) = Contrast;
 data_uji(k,10) = Homogeneity;
 
 %menyusun variabel data latih bentuk
 data_uji(k, 11) = Area;
 %data_latih(k, 12) = Perimeter;
 data_uji(k, 12) = Solidity;
 data_uji(k, 13) = Eccentricity;
 %data_latih(k, 15) = Metric;
 
end
%memanggil model naive bayes hasil pelatihan
load Mdl1
load Mdl_linear1
load Mdl_poly1
load Mdl_rbf1


% Menetapkan target_uji
    target_uji = cell(jumlah_file,1);
    for k = 1:30
        target_uji{k} = 'Setengah Matang';
    end
    
    for k = 31:60
        target_uji{k} = 'Matang';
    end
    
    for k = 61:90
        target_uji{k} = 'Busuk';
    end
    
    
% Prediksi kelas dengan model SVM
kelas_prediksi = predict(Mdl1, data_uji);

kelas_prediksilinear = predict (Mdl_linear1, data_uji);

kelas_prediksipoly = predict (Mdl_poly1, data_uji);

kelas_prediksirbf = predict (Mdl_rbf1, data_uji);

% Evaluasi kinerja model
CM = confusionmat(target_uji, kelas_prediksi);
accuracy = sum(diag(CM)) / sum(CM(:));

% Menampilkan hasil evaluasi
fprintf('Akurasi: %.2f%%\n', accuracy * 100);


save Mdl1 Mdl1
save Mdl_linear1 Mdl_linear1 
save Mdl_poly1 Mdl_poly1
save Mdl_rbf1 Mdl_rbf1