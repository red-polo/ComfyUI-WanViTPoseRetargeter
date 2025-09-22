# Optional helper; some ComfyUI setups auto-run install.py on startup.
# If not, users can run: python install.py
import os, sys, subprocess

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    req = os.path.join(here, 'requirements.txt')
    if os.path.exists(req):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req])
            print('[ComfyUI-WanViTPoseRetargeter] requirements installed.')
        except subprocess.CalledProcessError as e:
            print('[ComfyUI-WanViTPoseRetargeter] pip install failed:', e)
    else:
        print('[ComfyUI-WanViTPoseRetargeter] no requirements.txt found.')

if __name__ == '__main__':
    main()
