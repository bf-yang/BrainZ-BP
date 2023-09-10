# BrainZ-BP: A non-invasive cuff-less blood pressure estimation approach leveraging brain bio-impedance and electrocardiogram
- This is a repo for the paper: " A non-invasive cuff-less blood pressure estimation approach leveraging brain bio-impedance and electrocardiogram ".
- This is the first open-source **brain** bio-impedance dataset (BIOZ) for blood pressure (BP) estimation. We believe that this dataset will further contribute to the advancement of research in BP estimation using brain BIOZ, fostering progress in our community
- The database contains 1942 ECG and brain BIOZ recordings acquired from 13 subjects. Each 8-seconds recording has two labels (reference SBP and DBP). The mean values of systolic blood pressure (SBP) and diastolic blood pressure (DBP) in the database are recorded as 126.3 $\bm$ 14.6 mmHg and 73.3 ± 10.2 mmHg, respectively. $\sqrt{3x-1}+(1+x)^2$

# Dataset Structure
```
|--Data // Database for our approach BrainZ-BP  
  
|-- Data_seg_all // all the signal segments in our experiments  
|-- Excitation_freq // Different excitation frequency data  
    |-- anterior_posterior	// electrodes placed in anterior-posterior direction  
        |-- 1k      // excitation frequency 1 kHz  
         |-- 2k  
         |-- ...  
         |-- 20k  
      |-- left_right	// electrodes placed in left-right direction  
         |-- 1k  
         |-- 2k  
         |-- ...  
         |-- 20k  
   |-- s01_20211021 // subject 01 data, time:20211021  
   |-- s02_20211022 // subject 02 data, time:20211022   
   |-- s03_20211023 // subject 03 data, time:20211023  
   |-- s04_20211024 // subject 04 data, time:20211024   
   |-- ...  
   |-- s13_20211102 // subject 13 data, time:20211102   
  
|-- README.md  

