# Network Intrusion Detection using Deep Learning

Môn học: Học máy — LT01  
Dataset: UNSW-NB15

## Mô tả
Dự án xây dựng hệ thống phát hiện xâm nhập mạng (NIDS) sử dụng:
- Sequential DNN + Extra Tree Classifier (13 đặc trưng)
- 1D-CNN (43 đặc trưng)

## Kết quả
| Mô hình | Accuracy |
|---|---|
| DNN (13 features) | 99.57% |
| 1D-CNN (43 features) | 99.73% |

## Cách chạy
1. Tải dataset UNSW-NB15 tại: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2. Đặt file CSV vào cùng thư mục, đổi tên thành `UNSW_NB15_training-set.csv`
3. Cài thư viện: `pip install -r requirements.txt`
4. Chạy: `python main.py`

## Thành viên nhóm
- Nguyễn Lê Trung Khánh - 225748020110005
- Hà Linh Chi - 225748020110002
- Lê Thị Lan Hương - 225748020110034