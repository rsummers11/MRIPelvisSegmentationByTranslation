## **Instructions**  

#### Running the pipeline  
1. Prepare the data using the 'dataset' instructions below.
2. Convert nii.gz to .png files using the preprocessing instructions specified in Section 4.3 of the paper. 
3. We experimented 3 image synthesis models as shown below:    
   - Pix2Pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
   - CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))   
   - SynDiff (https://github.com/icon-lab/SynDiff/tree/main)   
4. Follow the data preparation procedures specified by each image translation model and train the image translation models. We use the default experiment setup to train the models.     
5. Repeat step 1. to prepare the testing data, and run the inference using the trained image synthesis model to generate synthetic CTs.    
6. Gather the 2D synthetic CT images generated by image-to-image translation models and reconstruct them into a 3D volume.   
7. Use TotalSegmentator to run the inference on the synthetic 3D volume to obtain the segmentation results.    
   - TotalSegmentator (https://github.com/wasserth/TotalSegmentator)    


#### Dataset  
1. Since we don't own the datasets used in the study, you need to download them. For training, we used Gold Atlas - Male Pelvis dataset while the CPTAC-UCEC for the testing. Please see the detailed information below to obtain the datasets.
2. Training: Gold Atlas - Male Pelvis dataset, which can be downloaded here: https://zenodo.org/records/583096  
3. Testing:  
       CPTAC-UCEC Dataset, which can be downloaded here: [https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19039602](https://www.cancerimagingarchive.net/collection/cptac-ucec/) . We used the T2 sequences as shown below. Then since the raw image files were in DICOM format, we used itk-snap to convert them to NIFTI format.  

          ├── CPTAC-UCEC\C3N-00858\01-08-2000-NA-MIEDNICA-65907\4.000000-t2bltsetra-76193  
          ├── CPTAC-UCEC\C3N-00860\01-22-2000-NA-MIEDNICA-33628\4.000000-t2bltsetra-29838  
          ├── CPTAC-UCEC\C3N-01001\02-23-2000-NA-MIEDNICA-97539\4.000000-t2bltsetra-55931  
          ├── CPTAC-UCEC\C3N-01003\02-19-2000-NA-MIEDNICA-66632\4.000000-t2bltsetra-09021    
          ├── CPTAC-UCEC\C3N-01007\04-01-2000-NA-MIEDNICA-78405\4.000000-t2bltsetra-13404    
          ├── CPTAC-UCEC\C3N-01172\05-04-2000-NA-MIEDNICA-78744\4.000000-t2bltsetra-03604    
          ├── CPTAC-UCEC\C3N-01341\05-09-2000-NA-BODYMIEDNICA-18754\5.000000-t2bltsetra-20976    
          ├── CPTAC-UCEC\C3N-01342\05-10-2000-NA-MIEDNICA-52074\4.000000-t2bltsetra-75036    
          ├── CPTAC-UCEC\C3N-01346\05-26-2000-NA-BODYMIEDNICA-60801\3.000000-t2bltsetra-10742    
          ├── CPTAC-UCEC\C3N-01761\06-08-2000-NA-MIEDNICA-08891\4.000000-t2bltsetra-02893      
          ├── CPTAC-UCEC\C3N-01763\06-01-2000-NA-MIEDNICA-68213\4.000000-t2bltsetra-02370      
          ├── CPTAC-UCEC\C3N-01764\06-06-2000-NA-BODYMIEDNICA-75164\4.000000-t2bltsetra-30211      
          ├── CPTAC-UCEC\C3N-01765\06-24-2000-NA-MIEDNICA-13084\4.000000-t2bltsetra-64390      
          ├── CPTAC-UCEC\C3N-01871\05-31-2000-NA-MIEDNICA-91535\4.000000-t2bltsetra-32087      
          ├── CPTAC-UCEC\C3N-01873\07-01-2000-NA-MIEDNICA-08363\4.000000-t2bltsetra-62871      
          ├── CPTAC-UCEC\C3N-01875\07-15-2000-NA-MIEDNICA-75940\4.000000-t2bltsetra-17941      
          ├── CPTAC-UCEC\C3N-01876\07-27-2000-NA-MIEDNICA-61790\4.000000-t2bltsetra-55722      
          ├── CPTAC-UCEC\C3N-01877\08-10-2000-NA-MIEDNICA-26947\4.000000-t2bltsetra-14823      
          ├── CPTAC-UCEC\C3N-01878\08-03-2000-NA-MIEDNICA-17690\4.000000-t2bltsetra-13491      
          ├── CPTAC-UCEC\C3N-01879\08-16-2000-NA-MIEDNICA-16519\4.000000-t2bltsetra-62254      
          ├── CPTAC-UCEC\C3N-01880\09-02-2000-NA-MIEDNICA-71707\4.000000-t2bltsetra-26489      
          ├── CPTAC-UCEC\C3N-02595\12-08-2000-NA-BODYMIEDNICA-36410\3.000000-t2bltsetra-18127     
          ├── CPTAC-UCEC\C3N-02631\12-09-2000-NA-MIEDNICA-37114\4.000000-t2bltsetra-07292      
          ├── CPTAC-UCEC\C3N-02632\11-08-2000-NA-MIEDNICA-08519\4.000000-t2bltsetra-40678      
          ├── CPTAC-UCEC\C3N-02639\01-04-2001-NA-MIEDNICA-14771\4.000000-t2bltsetra-00520      
          ├── CPTAC-UCEC\C3N-02678\12-29-2000-NA-BODYMIEDNICA-71292\3.000000-t2bltsetra-96765     
          ├── CPTAC-UCEC\C3N-02976\01-03-2001-NA-MIEDNICA-34482\4.000000-t2bltsetra-22047      
          ├── CPTAC-UCEC\C3N-02978\12-28-2000-NA-MIEDNICA-99235\4.000000-t2bltsetra-71537      
          ├── CPTAC-UCEC\C3N-02979\01-02-2001-NA-BODYMIEDNICA-43265\3.000000-t2bltsetra-51072      
          ├── CPTAC-UCEC\C3N-03417\04-05-2001-NA-MIEDNICA-82370\4.000000-t2bltsetra-39283    
		  
#### References
If you find our work is useful for your research, please consider citing
```bib
@article{zhuang2024segmentation,
  title={Segmentation of pelvic structures in T2 MRI via MR-to-CT synthesis},
  author={Zhuang, Yan and Mathai, Tejas Sudharshan and Mukherjee, Pritam and Summers, Ronald M},
  journal={Computerized Medical Imaging and Graphics},
  pages={102335},
  year={2024},
  publisher={Elsevier}
}
```
