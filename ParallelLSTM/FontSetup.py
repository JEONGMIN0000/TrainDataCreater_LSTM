
from IPython.core.magic import register_cell_magic
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont
import unicodedata
import warnings

%%skip installedKorFont 한글 폰트 설치되어 있으므로 재기동 하지 않음.
import os
import IPython

@register_cell_magic
def skip(line, cell):
    flag = line.strip().split(' ')[0]
    msg  = line.strip()[len(flag):]
    if globals()[flag] == True:
        print(msg)
        return
    else:
        exec(cell, globals())


def restart_kernel():
    IPython.Application.instance().kernel.do_shutdown(restart=True)

restart_kernel()
print("Kernel is restarted.")


def check_korean_font_available():
    # 한글 유니코드 범위
    def is_korean(char):
        return 0xAC00 <= ord(char) <= 0xD7A3 or 0x3131 <= ord(char) <= 0x3163

    # 모든 폰트를 순회
    for font_path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            font = TTFont(font_path)
            for cmap in font['cmap'].tables:
                if cmap.isUnicode():
                    for char in cmap.cmap.keys():
                        if is_korean(chr(char)):
                            return True
        except:
            continue
    
    return False

# 한글 폰트 사용 가능 여부 : True
installedKorFont = check_korean_font_available()
print(f"- 한글 폰트 사용 가능: {installedKorFont}")

# Google Colab에서 실행 여부 : False
isColab = 'google.colab' in str(get_ipython())
print(f"- Google Colab 여부 : {isColab}")


# 환경 설정
if not installedKorFont:
    install_fonts()

#한글 폰트 관련 warning supress
warnings.filterwarnings(action='ignore')

#fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
fontpath = './font/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)

%matplotlib inline

# 그래프에 retina display 적용
%config InlineBackend.figure_format = 'retina'

# Colab 의 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic')


# jupyter nbconvert 실행시 한글이 나오지 않는 경우 아래 comment 해제 
#fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf')
fm.fontManager.addfont(fontpath)

# 한글 출력 테스트

plt.text(0.3, 0.4, '한글', size=80)