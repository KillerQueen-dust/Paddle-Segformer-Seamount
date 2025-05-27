# PaddleSeg Usage Guide

This repository provides a setup and prediction example for using PaddleSeg.

## 📦 Installation

Clone the repository and install the dependencies:

```bash
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

## 📥 Download Pretrained Model

Download the pretrained model file from the following link:

👉 [https://pan.quark.cn/s/344bfd44dd1d](https://pan.quark.cn/s/344bfd44dd1d)

After downloading, place the model file into the **root directory** of this repository.

## 🧪 Run Prediction

To run the prediction example:

```bash
sh predict.sh
```

This script will use the downloaded model to perform inference on the provided test case.

---

Feel free to open an issue if you encounter any problems during setup or testing.