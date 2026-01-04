🧬 Protein Function Prediction using Deep Learning
📌 Project Overview

Proteins are fundamental biomolecules responsible for nearly all biological processes. However, a large number of newly discovered protein sequences remain functionally uncharacterized. Traditional experimental methods for protein function identification are costly, time-consuming, and labor-intensive.

This project presents a deep learning–based system that predicts the molecular function of proteins directly from their amino acid sequences. The model leverages a combination of Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks to automatically learn meaningful biological patterns from raw sequence data.

The predicted functions are expressed as Gene Ontology (GO) molecular function terms, making the output biologically interpretable and scientifically useful.

🧠 Model Architecture

The proposed model integrates the strengths of multiple neural network components:

Component	Purpose
Embedding Layer	Converts amino acid indices into dense vector representations
CNN (Conv1D)	Extracts local motifs and conserved patterns
BiLSTM	Captures long-range dependencies and full sequence context
Global Average Pooling	Reduces dimensionality while preserving important features
Fully Connected Layers	Perform final multi-label classification
Sigmoid Activation	Produces probability scores for each GO term
🧪 Dataset

Protein sequences and their associated functional annotations are collected from public biological databases.
Each protein sequence is encoded into numeric format and padded to a fixed length for uniform processing.

⚙️ Training Details

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Regularization: Dropout, L2 Regularization

Handling Class Imbalance: Imbalanced-learn techniques

Evaluation Metrics:

Precision, Recall, F1-Score (Micro, Macro, Weighted)

ROC-AUC

Training & Validation Accuracy

Training & Validation Loss

📊 Performance Visualization

The training process is monitored using:

Accuracy Curves

Loss Curves

ROC Curve

These curves help assess convergence, generalization, and classification quality.

🧾 Output Interpretation

For any input protein sequence, the system returns the Top-K predicted Gene Ontology functions with their confidence scores:

GO:0000257 (nitrilase activity) → Confidence: 0.264
GO:0000463 (maturation of LSU-rRNA...) → Confidence: 0.204
...


This provides both biological meaning and quantitative confidence for each prediction.

🧑‍💻 How to Run
1️⃣ Install Dependencies
pip install tensorflow scikit-learn imbalanced-learn numpy pandas matplotlib

2️⃣ Run Training
python train.py

3️⃣ Predict Protein Function

Provide a protein sequence:

test_protein = "MKTLLILTCLVAVALARPK..."


Run the prediction cell to obtain GO term predictions.

🎯 Applications

Functional annotation of novel proteins

Accelerating biological research

Drug discovery & protein engineering

Genome analysis and annotation pipelines

🏁 Conclusion

This project demonstrates that deep learning models combining CNN and BiLSTM architectures can effectively learn complex biological patterns from protein sequences and produce reliable functional predictions. The approach reduces dependency on manual feature engineering and significantly accelerates protein function annotation.
