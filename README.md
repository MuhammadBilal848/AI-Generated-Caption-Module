# AI-Generated-Caption-Module
Contains a captioning module that takes videos and generates text-based captions.

## Installation

1. First clone the repository or download the zip by clicking on the code dropdown.
2. Open **Anaconda Prompt Powershell** or **Anaconda Prompt (Anaconda3)** from your startup. It should show something like this: 
```bash
(base) PS C:\Users\User_Name >
```

3. Change the directory by copying and pasting the folder path where you cloned the repository by using the following command:
```bash
(base) PS C:\Users\User_Name > cd cloned_repo_path
```

4. Open the repository in **Vs Code** by using the following command:
```bash
(base) PS X:\cloned_repo_path > code .
```

5. Open **Bash** or **Powershell** in **Vs Code**.

6. Create a conda environment by using the following command:
```bash
conda create -p tezeract python==3.10 -y
```    

7. Activate the conda environment by using the following command:
```bash
conda activate tezeract/
```    

8. Install all the necessary packages by using the following command:
```bash
pip install -r requirements.txt
```    

9. Now run **Flask Application** by running this command:
```bash
python app.py
```
**OR**
```
uvicorn app:app
```
