#!/bin/bash
wget -O model_300.pt https://www.dropbox.com/s/opwm4dg0vxcq6p4/model_300.pt?dl=1
wget -O model_generator_90.pt https://www.dropbox.com/s/fcz94x74o8txq9f/model_generator_90.pt?dl=1
wget -O model_acgan_generator_180.pt https://www.dropbox.com/s/jpilzg7yf9qtkor/model_acgan_generator_180.pt?dl=1
python3 plot_data/plot_1_2.py plot_data/vae_mse.csv plot_data/vae_kld.csv 1000 per_400_batch mse vae_mse kld vae_kld
python3 plot_data/plot_2_2.py plot_data/2-2_d_real_loss_curve.csv plot_data/2-2_d_fake_loss_curve.csv plot_data/2-2_dis_accu_real.csv plot_data/2-2_dis_accu_fake.csv BCE_loss Accuracy_of_discriminator
python3 plot_data/plot_3_2.py plot_data/3-2_d_real_label_loss_curve.csv plot_data/3-2_d_fake_label_loss_curve.csv plot_data/3-2_d_accu_real_curve.csv plot_data/3-2_d_accu_fake_curve.csv Training_Loss_of_Attribute_Classification Accuracy_of_discriminator
python3 fig1_3.py -b 16 -e 400 -tn 40000 -lat 1024 -lam 0.0001 -tm 40000 -exp plotplot -tsne 4 -td $1/test
python3 fig1_4.py -b 16 -e 400 -tn 40000 -lat 1024 -lam 0.0001 -tm 40000 -exp plotplot -tsne 4
python3 tsne.py  -tn 40000 -lat 1024 -tm 1024 -b 16 -ra 1
python3 acgan.py -b 16 -e 400 -tn 40000 -lat 101 -lam 0.0001 -tm 40000 -exp plotplot
python3 dcgan.py -b 16 -e 400 -tn 40000 -lat 128 -lam 0.0001 -tm 40000 -exp plotplot
python3 plot_data/plot_1_2.py plot_data/vae_mse.csv plot_data/vae_kld.csv 1000 per_400_batch mse vae_mse kld vae_kld
python3 plot_data/plot_2_2.py plot_data/2-2_d_real_loss_curve.csv plot_data/2-2_d_fake_loss_curve.csv plot_data/2-2_dis_accu_real.csv plot_data/2-2_dis_accu_fake.csv BCE_loss Accuracy_of_discriminator
python3 plot_data/plot_3_2.py plot_data/3-2_d_real_label_loss_curve.csv plot_data/3-2_d_fake_label_loss_curve.csv plot_data/3-2_d_accu_real_curve.csv plot_data/3-2_d_accu_fake_curve.csv Training_Loss_of_Attribute_Classification Accuracy_of_discriminator
python3 fig1_3.py -b 16 -e 400 -tn 40000 -lat 1024 -lam 0.0001 -tm 40000 -exp plotplot -tsne 4 
python3 fig1_4.py -b 16 -e 400 -tn 40000 -lat 1024 -lam 0.0001 -tm 40000 -exp plotplot -tsne 4
python3 tsne.py  -tn 40000 -lat 1024 -tm 1024 -b 16 -ra 1
python3 acgan.py -b 16 -e 400 -tn 40000 -lat 101 -lam 0.0001 -tm 40000 -exp plotplot
python3 dcgan.py -b 16 -e 400 -tn 40000 -lat 128 -lam 0.0001 -tm 40000 -exp plotplot






























