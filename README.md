# BrainZ-BP: A non-invasive cuff-less blood pressure estimation approach leveraging brain bio-impedance and electrocardiogram
This is a repo for the paper: " A non-invasive cuff-less blood pressure estimation approach leveraging brain bio-impedance and electrocardiogram ".


Project Structure

|--sample-code-UTD // sample code of each approach on the UTD dataset

 |-- Cosmo                    // codes of our approach Cosmo
    |-- main_con.py/	// main file of contrastive fusion learning on cloud 
    |-- main_linear_iterative.py/	// main file of supervised learning on edge
    |-- data_pre.py/		// prepare for the data
    |-- cosmo_design.py/ 	// fusion-based feature augmentation and contrastive fusion loss
    |-- cosmo_model_guide.py/	// models of unimodal feature encoders and the quality-quided attention based classifier
    |-- util.py/	// utility functions

 |-- CMC                    // codes of the baseline Contrastive Multi-view Learning (CMC)
    |-- main_con.py/	// main file of contrastive multi-view learning
    |-- main_linear_iterative.py/	// main file of supervised learning
    |-- data_pre.py/		// prepare for the data
    |-- cosmo_design.py/ 	//  feature augmentation and contrastive loss
    |-- cosmo_model_guide.py/	// models of unimodal feature encoders and the attention based classifier
    |-- util.py/	// utility functions

 |-- supervised-baselines                    // codes of the supervised learning baselines
    |-- attnsense_main_ce.py/	// main file of AttnSense
    |-- attnsense_model.py/	// models of AttnSense
    |-- deepsense_main_ce.py/	// main file of DeepSense
    |-- deepsense_model.py/	// models of DeepSense
    |-- single_main_ce.py/	// main file of single modal learning
    |-- single_model.py/	// models of single modal learning (IMU and skeleton)
    |-- data_pre.py/		// prepare for the multimodal and singlemodal data
    |-- util.py/	// utility functions
    
|--UTD-data 	// zipped folder for the processed and splited data in the UTD dataset

|-- README.md

|-- materials               // figures and materials used this README.md
