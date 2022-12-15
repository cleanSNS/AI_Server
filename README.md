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
 
### 필요한 파일 다운로드
다음 파일들을 다운 받아서 동일한 폴더에 배치

#### 인공지능 weight 다운로드
- 002/hi : Swear Word Detector
https://drive.google.com/drive/folders/1--rBur0qpMURVvbnKJ_yREaLeEQlxgEs?usp=share_link

- sentence_classifier/001 : Sentence Classifier
https://drive.google.com/drive/folders/1Ue9TGgZ1z0kZPGv-IR7Crx2wq9Z-a7Xq?usp=share_link


### colab
#### Swear Word Detector
문장의 각 토큰이 욕설인지 아닌 지 확인
https://colab.research.google.com/drive/1y8vJg8gvlv86r4fipOX0ERUsqaLRmiaz?usp=sharing

**dataset 다운로드**

https://drive.google.com/file/d/15EV3_mahFce17TMmJ_KgIzdFFcHKiLDT/view?usp=share_link

#### Sentence Classifier
문장에 욕설이 있는 지 확인
https://colab.research.google.com/drive/1CWRCv5zvGI-AT1XWMYX-iurVVtpDHpOq?usp=sharing

**dataset 다운로드**

https://drive.google.com/file/d/1QZ-_IZ9td2WErFQlucoqsi5TjPRBvL5h/view?usp=sharing
https://drive.google.com/file/d/1BkaEXOYVF4sQMCSnZFvxJvnbyFvjzMi9/view?usp=sharing
https://drive.google.com/file/d/1iq8bKboFybBl_vTs_5aqbz1_gkyOyHqR/view?usp=sharing
https://drive.google.com/file/d/1bnDOdBi45bWzIaEQyxbz-GtDDDN20Mcm/view?usp=sharing

### 서버 실행
- ai.py : swear word detector만 존재하는 서버파일
- ai2.py : swear word detector + sentence classifier가 존재하는 서버파일

```
uvicorn <파일 명>:app --reload --host=0.0.0.0 --port=8000

# uvicorn ai:app --reload --host=0.0.0.0 --port=8000
```
