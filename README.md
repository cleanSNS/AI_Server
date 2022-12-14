### 주의사항
ec2(t2.medium 이상)에서만 동작합니다. 

### python 가상환경 구성
sudo apt install python3.7 python3-venv python3.7-venv

python3.7 -m venv <폴더 명>

### 가상환경 실행
. <폴더명>/bin/activate

python --version
>> 3.7.15

### 필요한 라이브러리 설치

pip install -r requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
 
### 인공지능 weight 다운로드
https://drive.google.com/drive/folders/1--rBur0qpMURVvbnKJ_yREaLeEQlxgEs?usp=share_link



### 서버 실행
```
uvicorn <파일 명>:app --reload --host=0.0.0.0 --port=8000
```
