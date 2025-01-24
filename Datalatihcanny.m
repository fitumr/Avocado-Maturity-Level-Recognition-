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
% Deteksi tepi menggunakan metode Canny
canny_edge = edge(img, 'Canny');

 % Ekstraksi fitur bentuk
    stats = regionprops(canny_edge, 'Area', 'Perimeter', 'Solidity', 'Eccentricity');

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
        target_latih{k} = 'Mentah';
    end
    
    for k = 81:160
        target_latih{k} = 'Matang';
    end
    
    for k = 161:240
        target_latih{k} = 'Busuk';
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