# CancerModel
This repository contains code pertaining to the tumor transport model and its application in PSOR's recent paper **Optimal Therapy Design with Tumor Microenviornment normalization** [1]. The tumor transport model was originally developed by Baxter and Jain [2-5] and refined in Martin et al. [6]. The model was reimplemented in Julia for the work of Wang et al. [1] and eventually implemented in ModelingToolkit.jl [7] using MethodOfLines.jl.

## Organization
- ### Parameter Estimation Problems
  This folder contains code pertaining to the parameter estimation problems solved in Wang et al. [1]. It contains 4 Jupyter notebooks corresponding to each of the 4 ANN surrogate models developed. The notebooks export the ANN surrogate models as .BSON files for use in optimization. Additionally, this folder contains the PE_master.jl file used to perform parameter estimation files and the corresponding utily file surrogate_optimization_functions.jl.
- ### Drug and Therapy Design Problems
  This folder contains code pertaining to the simultaneous drug and therapy design problems solved in Wang et al. [1]. It contains two subfolders- the 3 parameter problem subfolder and the modified problem subfolder. Each folder contains two notebooks to train and export the average concentration model and the peak concentration model as well as a file TreatmentANN.jl file to solve the simultaneous drug and therapy design optimization problem. The 3 parameter models takes in vascular hyraulic conductivity (Lp), interstitial hydraulic conductivity (K), and nanoparticle size (rs) and require an external routine for mapping therapeutic dosage to Lp and K. The modified models take in therapeutic dosage and nanoparticle size with an embdedded regresion model for mapping from Lp and K to the concentration. The 3 parameter models are more flexible and were the models ultimately used in the paper.
- ### ModelingToolkit Implementation
  The CancerModelMTK.jl file contains an updated tumor transport model implemented in ModelingToolkit.jl [7] using MethodOfLines.jl. The model utilizes the analytical solution to pressure. The authors strongly suggest any future applications of this work should utilize the ModelingToolkit model instead of the vanilla Julia model.
- ### Supplementary Documentation
  A thorough description of the tumor transport model, machine learning surrogate, and optimization problems is documented in Samuel Degnan-Morgenstern's undergraduate thesis included in the supplementary documentation folder.

## References
1. Wang, C., Degnan‐Morgenstern, S., Martin, J. D., & Stuber, M. D. **Optimal Therapy Design With Tumor Microenvironment Normalization.** *AIChE Journal*, e17747.
2. Baxter, L. T., & Jain, R. K. (1989). **Transport of fluid and macromolecules in tumors. I. Role of interstitial pressure and convection.** *Microvascular research*, 37(1), 77-104.
3. Baxter, L. T., & Jain, R. K. (1990). **Transport of fluid and macromolecules in tumors. II. Role of heterogeneous perfusion and lymphatics.** *Microvascular research*, 40(2), 246-263.
4. Baxter, L. T., & Jain, R. K. (1991). **Transport of fluid and macromolecules in tumors: III. Role of binding and metabolism.** *Microvascular research*, 41(1), 5-23.
5. Baxter, L. T., & Jain, R. K. (1991). **Transport of fluid and macromolecules in tumors. IV. A microscopic model of the perivascular distribution.** *Microvascular research*, 41(2), 252-272.
6. Martin, J. D., Panagi, M., Wang, C., Khan, T. T., Martin, M. R., Voutouri, C., Toh, K., Papageorgis, P., Mpekris, F., Polydorou, C., Ishii, G., Takahashi, S., Gotohda, N., Suzuki, T., Wilhelm, M. E., Melo, V. A., Quader, S., Norimatsu, J., Lanning, R. M., Kojima, M., Stuber, M. D., Stylianopoulos, T., Kataoka, K., and Cabral, H. **Dexamethasone Increases Cisplatin-Loaded Nanocarrier Delivery and Efficacy in Metastatic Breast Cancer by Normalizing the Tumor Microenvironment.** *ACS Nano.* 13(6), 6396-6408 (2019).
7. Ma, Y., Gowda, S., Anantharaman, R., Laughman, C., Shah, V., & Rackauckas, C. (2021). **Modelingtoolkit: A composable graph transformation system for equation-based modeling.** *arXiv preprint* arXiv:2103.05244.
## Contact

Samuel Degnan-Morgenstern, Former Undergraduate Researcher at the University of Connecticut / Current PhD student at MIT, samuel.morgenstern@uconn.edu / stm16109@mit.edu

Dr. Matthew Stuber, Principal Investigator, matthew.stuber@uconn.edu
