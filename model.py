"""
Quantum BioNet 2.0
This repository accompanies the published paper and is provided for
academic inspection and educational purposes.

NOTE:
- Google Colab-specific code has been removed.
- Dataset paths and pretrained weights are intentionally omitted.
- Additional resources are required to reproduce the published results.
"""


# MOUNT GOOGLE DRIVE
import zipfile
import os
import shutil

zip_path = "YOUR_DATASET_PATH"
extract_to = "YOUR_DATASET_PATH"

if os.path.exists(extract_to):
    shutil.rmtree(extract_to)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Top-level folders/files after extraction:")
for item in os.listdir(extract_to):
    print(item)

# ---- Section ----

# 6️⃣ PennyLane Quantum Circuit & TorchLayer
dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS), ranges=(3, 3, 3))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
weight_shapes = {"weights": (3, N_QUBITS, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ---- Section ----

# 7️⃣ Hybrid Model Definition
class HybridNet(nn.Module):
    def __init__(self, num_classes, stage):
        super().__init__()
        self.feature_extractor = models.resnet152(weights='IMAGENET1K_V2')
        # Fine-tune last blocks for best accuracy
        for param in list(self.feature_extractor.parameters())[:-40]:
            param.requires_grad = False
        self.feature_extractor.fc = nn.Identity()

        self.cnn_stream = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.q_dense = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N_QUBITS),
            nn.BatchNorm1d(N_QUBITS),
            nn.ReLU()
        )
        self.qlayer = qlayer

        self.fusion = nn.Sequential(
            nn.Linear(512 + N_QUBITS, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.out = nn.Linear(256, 1 if stage==1 else num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        c_out = self.cnn_stream(feats)
        q_in = self.q_dense(feats)
        q_out = self.qlayer(q_in)
        fused = torch.cat([c_out, q_out], dim=1)
        fused = self.fusion(fused)
        return self.out(fused)