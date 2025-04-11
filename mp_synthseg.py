import multiprocessing
import subprocess
import glob
import os

dst_dir = '/root/data/ADNIz_synthseg_out'

def run_mri_synthseg(input_file):
    try:
        # mri_synthseg 명령어 실행
        command = [
            'mri_synthseg',
            '--i', input_file,
            '--o', input_file,
            '--robust',
            '--vol', f'{input_file}/vol',
            '--threads', '20'
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"{input_file} 처리 완료:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {input_file}: {e}")

if __name__ == '__main__':
    files = glob.glob('/root/sexDiff/samples/*') #, recursive=True)
    print(len(files))
    # exit()
    # 멀티프로세싱을 위한 Pool 생성 (4개의 코어 사용)
    with multiprocessing.Pool(processes=5) as pool:
        # 각 파일에 대해 mri_synthseg 병렬로 실행
        pool.map(run_mri_synthseg, files)
