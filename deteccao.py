#%%

import cv2
import numpy as np
import pandas as pd

path = 'C:/Users/t-pim/OneDrive/UFES/TCC/Video Analytics/videos/'

#coordenadas das chamines
df_chamines_base = pd.read_excel(path+'mapas.xlsx', sheet_name='chamines')
#detalhes das emissoes reais
df_emissoes = pd.read_excel(path+'mapas.xlsx', sheet_name='emissoes')
#lista de videos e coordenadas do ROI
df_calibracao = pd.read_excel(path+'mapas.xlsx', sheet_name='calibracao')

#%%

#função que calcula o threshold dinâminco     
def adaptive_thresh(frame):
    global last_frame

    x0 = int(last_frame.shape[1]/2)
    y0 = int(last_frame.shape[0]/2)

    x0 = 90
    y0 = 590

    roi = frame[y0-10:y0+10, x0-10:x0+10]

    thresh_value = int(roi.mean()*(FRACAO/100))

    return thresh_value

#funcao para transformar o frame em preto e branco sem ruidos
def clean_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    hist = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(hist, (5, 5), 0)
    thresh_value = adaptive_thresh(blur)
    ret, thresh = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening 

#funcao para achar a posicao de estabilidade do frame devido a movimentacao da camera
#seleciona uma parte da imagem do primeiro frame e procura sua posicao correspondente do frame atual
#calcula a diferenca das posicoes entre os frames e enquadra o frame atual
#subtrai o primeiro frame do frame atual para eliminar os objetos fixos (sombra do porto)
def estabilizador(frame, last_frame):

    x0 = 90
    y0 = 590
    x1 = 150
    y1 = 660

    delta = 5

    last_frame_clean = clean_image(last_frame)
    base = last_frame_clean[y0+delta:y1-delta, x0+delta:x1-delta] 

    frame_clean = clean_image(frame)
    compare = frame_clean[y0:y1, x0:x1]

    #methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    result = cv2.matchTemplate(base, compare, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    dx=max_loc[0]-delta
    dy=max_loc[1]-delta-5

    roi_last = last_frame_clean[upper_left[1]-dy:bottom_right[1]-dy, upper_left[0]-dx:bottom_right[0]-dx]
    roi = frame_clean[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    kernel = np.ones((3,7))
    dilate = cv2.dilate(roi_last, kernel, iterations=1)

    diff = roi - dilate

    #acerta valores diferentes de 0 e 255 que aparecem na subtração
    ret, thresh = cv2.threshold(diff, 254, 255, cv2.THRESH_BINARY)

    return thresh


#%%

df_chamines = df_chamines_base.copy()
df_chamines['y0'] = np.zeros((df_chamines.shape[0],1)).astype(int)

df_chamines['emissao'] = np.zeros((df_chamines.shape[0],1)).astype(int)
df_chamines['tempo'] = np.zeros((df_chamines.shape[0],1)).astype(int)
df_chamines['break'] = np.zeros((df_chamines.shape[0],1)).astype(int)
df_chamines['block'] = np.zeros((df_chamines.shape[0],1)).astype(int)
df_chamines['blocked'] = np.zeros((df_chamines.shape[0],1)).astype(int)

df_chamines['block']=df_chamines['block'].astype('object')
df_chamines['blocked']=df_chamines['blocked'].astype('object')

df_mapa = df_chamines.loc[:,['chamine','x0','x1','y0','y1']].copy()

#tabela para armazenar dados das emissoes detectadas
df_deteccoes = pd.DataFrame(columns=['chamine','frame_inicio','frame_fim','duracao_frame'])

file_num = 0
display(f'VIDEO {file_num}')
#dados do video que sera aberto
file_name = df_calibracao.loc[file_num,'file']

#coordenadas de enquadramento do video aberto
top = df_calibracao.loc[0,'y0']
bottom = df_calibracao.loc[0,'y1']
left = df_calibracao.loc[0,'x0']
right = df_calibracao.loc[0,'x1']

upper_left = (int(left), int(top))
bottom_right = (int(right), int(bottom))

cap = cv2.VideoCapture(path+file_name+'.mp4')

if (cap.isOpened()== False):
    print("Não foi possível ler o arquivo")
    exit()

ret, last_frame = cap.read()
if ret is False:
    print("Erro ao ler o stream do video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#numero maximo de frames sem deteccar de emissao para finalizar a deteccao
BREAK_MAX=int(2*fps)
FRACAO = 55

#%%
while(1):
    ret, frame = cap.read()
    if ret is False:
        break

    #pre-processamento
    diff = estabilizador(frame, last_frame)

    #percorre as chamines para detectar emissoes
    for i in range(df_chamines.shape[0]):
        if df_chamines.loc[i,'blocked']==0:
            x0 = int(df_chamines.loc[i,'x0'])
            x1 = int(df_chamines.loc[i,'x1'])
            y0 = int(df_chamines.loc[i,'y0'])
            y1 = int(df_chamines.loc[i,'y1'])

            roi = diff[y0:y1, x0:x1]
            contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #registra dados da emissão
            if len(contours)>0:
                if df_chamines.loc[i,'emissao']==0:
                    df_chamines.at[i,'emissao'] = 1
                df_chamines.at[i,'tempo'] = df_chamines.loc[i,'tempo']+1
                df_chamines.at[i,'break'] = 0
            elif df_chamines.loc[i,'tempo'] > 0 :
                df_chamines.at[i,'tempo'] = df_chamines.loc[i,'tempo']+1
                if df_chamines.loc[16,'emissao']== 1 and i==12:
                    pass
                else:
                    df_chamines.at[i,'break'] = df_chamines.loc[i,'break']+1

            #acompanha emissão
            if len(contours)>0:
                while(True):
                    go_left = False
                    go_right = False
                    go_up = False

                    for k in range(len(contours)):

                        c = contours[k]
                        extLeft = tuple(c[c[:, :, 0].argmin()][0])
                        extRight = tuple(c[c[:, :, 0].argmax()][0])
                        extTop = tuple(c[c[:, :, 1].argmin()][0])

                        if extLeft[0] == 0:
                            go_left = True                            
                        if extRight[0] == (x1-x0)-1:
                            go_right = True
                        if extTop[1] == 0:
                            go_up = True

                    if not go_left and not go_right and not go_up:
                        break 
    
                    if go_left:
                        x0 = x0 - 5
                    if go_right:
                        x1 = x1 + 5
                    if go_up:
                        y0 = y0 - 5

                    if go_left or go_right or go_up:
                        roi = diff[y0:y1, x0:x1]
                        contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                df_mapa.at[i,'x0'] = x0
                df_mapa.at[i,'x1'] = x1
                df_mapa.at[i,'y0'] = y0
                
                #block vizinhos
                for k in range(df_chamines.shape[0]):
                    x0_vizinho = int(df_chamines.loc[k,'x0'])
                    x1_vizinho = int(df_chamines.loc[k,'x1'])
                    if  (x0_vizinho < x0 and x0 < x1_vizinho) or (x0_vizinho < x1 and x1 < x1_vizinho):
                        if df_chamines.loc[k,'blocked'] == 0:
                            df_chamines.at[k,'blocked'] = []
                        lista = df_chamines.loc[k,'blocked']
                        if i not in lista:
                            lista.append(i)
                        df_chamines.at[k,'blocked'] = lista

                        if df_chamines.loc[i,'block'] == 0:
                            df_chamines.at[i,'block'] = []
                        lista = df_chamines.loc[i,'block']
                        if k not in lista:
                            lista.append(k)
                        df_chamines.at[i,'block'] = lista
    #sinaliza emissões
    lista_emissoes = df_chamines[df_chamines['emissao']==1].index
    if len(lista_emissoes)>0:
        qnt_emissoes = len(lista_emissoes)
        cv2.rectangle(frame, pt1=(10, 10), pt2=(360, 120+((qnt_emissoes-1)*65)), color=(0, 0, 255), thickness=-1)
        cv2.putText(frame, text='Emissao detectada', org=(20,50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

        text_y1=85
        text_y2=110
        delta_y=text_y2-text_y1
        delta_text=40

        for i in lista_emissoes:
            x0 = int(df_chamines.loc[i,'x0'])
            x1 = int(df_chamines.loc[i,'x1'])
            y0 = int(df_chamines.loc[i,'y0'])
            y1 = int(df_chamines.loc[i,'y1'])

            delta = x1-x0
            x0_arrow = int(x0 + delta/2)
            x1_arrow = x0_arrow
            y0_arrow = y1

            if i <12:
                y1_arrow = 800
            if i >=12:
                y0_arrow = 800
                y1_arrow = 900

            x0 = x0 + upper_left[0] - 100
            x1 = x1 + upper_left[0] + 100
            y0 = y0 + upper_left[1] - 150 
            y1 = y1 + upper_left[1] + 50

            if len(df_chamines.loc[i,'chamine']) == 3:
                sub_texto = 25
            elif len(df_chamines.loc[i,'chamine']) == 4:
                sub_texto = 33
            elif len(df_chamines.loc[i,'chamine']) == 5:
                sub_texto = 40

            cv2.arrowedLine(frame, pt1=(x1_arrow, y1_arrow), pt2=(x0_arrow, y0_arrow), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, text=df_chamines.loc[i,'chamine'], org=(int(x0_arrow-sub_texto),y1_arrow+30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)

            cv2.putText(frame, text='Chamine:', org=(20,text_y1), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(frame, text=df_chamines.loc[i,'chamine'], org=(150,text_y1), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(frame, text='Duracao:', org=(20,text_y2), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(frame, text=str(int(df_chamines.loc[i,'tempo']/fps)), org=(150,text_y2), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
            
            text_y1=text_y2+delta_text
            text_y2=text_y1+delta_y

    #finalizar emissão
    lista_break = df_chamines[df_chamines['break']>BREAK_MAX].index
    for i in lista_break:
        final = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-df_chamines.loc[i,'break']
        duracao = df_chamines.loc[i,'tempo']-df_chamines.loc[i,'break']
        df_deteccoes = df_deteccoes.append({'chamine': df_chamines.loc[i,'chamine'],
                            'frame_inicio': (final-duracao),
                            'frame_fim': final,
                            'duracao_frame': duracao}, 
                            ignore_index=True)
        lista_block = df_chamines.loc[i,'block']
        if not lista_block == 0:
            for k in lista_block:
                lista = df_chamines.loc[k,'blocked']
                lista.remove(i)
                if len(lista)==0:
                    lista = 0
                df_chamines.at[k,'blocked'] = lista
        df_chamines.at[i,'block'] = 0
        df_chamines.at[i,'emissao'] = 0
        df_chamines.at[i,'tempo'] = 0
        df_chamines.at[i,'break'] = 0
        df_mapa.at[i,'x0'] = df_chamines.loc[i,'x0']
        df_mapa.at[i,'x1'] = df_chamines.loc[i,'x1']
        df_mapa.at[i,'y0'] = df_chamines.loc[i,'y0']

    cv2.imshow('window',frame)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

display(df_deteccoes)
display(df_emissoes[df_emissoes['file']==file_name])
