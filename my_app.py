import streamlit as st
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
#import playsound
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pygame


# Function to play an alarm
def tocar_alarme(caminho):
    pygame.mixer.init()
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()

# Function to calculate EAR
def calcular_ear(olho):
    # Calcula as distâncias euclidianas entre os dois conjuntos de
    # marcos oculares verticais (coordenadas x, y)
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])

    # Calcula a distância euclidiana entre os marcos oculares horizontais
    # (coordenadas x, y)
    C = dist.euclidean(olho[0], olho[3])

    # Calcula o EAR
    ear = (A + B) / (2.0 * C)

    # Retorna o EAR
    return ear

# Inicialize st.session_state.running e st.session_state.refresh_pressed se eles ainda não estiverem definidos
if "running" not in st.session_state:
    st.session_state.running = False
if "refresh_pressed" not in st.session_state:
    st.session_state.refresh_pressed = False

# Streamlit settings
st.title("Drowsiness Detector")
st.sidebar.title("Settings")

# Replace argparse with Streamlit's input methods
alarme = st.sidebar.checkbox("Usar alarme sonoro?", value=True)
#webcam = st.sidebar.slider("índice da webcam no sistema", 0, 4, 0)
webcam = st.sidebar.selectbox("Selecione o índice da webcam:", [0, 1, 2, 3, 4])


# Define constants and initializations
LIMIAR_EAR = 0.25
QTD_CONSEC_FRAMES = 20
CONTADOR = 0
ALARME_ON = False

print("[INFO] Carregando preditor de marcos faciais...")
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(inicio_esq, fim_esq) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(inicio_dir, fim_dir) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] Iniciando thread de fluxo de vídeo...")
vs = VideoStream(src=webcam).start()
time.sleep(1.0)

# Streamlit placeholders for video and graph
frame_slot = st.empty()
graph_slot = st.empty()


# Initialize the figure for plotting
y = [None] * 100
x = np.arange(0,100)
fig, ax = plt.subplots(figsize=(4, 2))

# Estilo e contexto
sns.set_style("whitegrid")
sns.set_context("notebook")

# Cores
colors = sns.color_palette("viridis", n_colors=1)
li, = ax.plot(x, y, color=colors[0], label="EAR")

# Título e rótulos
ax.set_title("Monitoramento do EAR ao longo do tempo")
ax.set_xlabel("Frames")
ax.set_ylabel("EAR")

# Limites e grades
ax.set_xlim([0, 100])
ax.set_ylim([0, 0.4])

# Legenda
ax.legend()

# Crie o botão "Parar" fora do loop
#botao_parar = st.button("Parar")

# Inicialize o contador de iterações
iter_count = 0

# Defina uma taxa de atualização (a cada quantos frames o vídeo será atualizado)
TAXA_ATUALIZACAO = 5

# Inicialize o contador de frames
contador_frames = 0


# Crie um espaço reservado para o botão "Iniciar"
start_button_placeholder = st.empty()

# Se o botão "Refresh" for pressionado, defina st.session_state.running como False
if st.button("Refresh", key="refresh_button"):
    st.session_state.running = False

# Exiba o botão "Iniciar" apenas se o programa não estiver rodando
if not st.session_state.running:
    if start_button_placeholder.button("Iniciar", key="start_button"):
        st.session_state.running = True

# Loop over the video frames
while st.session_state.running:
    # Limpe o espaço reservado do botão "Iniciar" para que ele desapareça durante a execução
    start_button_placeholder.empty()
    
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = preditor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        olho_esq = shape[inicio_esq:fim_esq]
        olho_dir = shape[inicio_dir:fim_dir]
        ear_esq = calcular_ear(olho_esq)
        ear_dir = calcular_ear(olho_dir)
        ear = (ear_esq + ear_dir) / 2.0

        casco_olho_esq = cv2.convexHull(olho_esq)
        casco_olho_dir = cv2.convexHull(olho_dir)
        cv2.drawContours(frame, [casco_olho_esq], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [casco_olho_dir], -1, (0, 255, 0), 1)

        y.pop(0)
        y.append(ear)

        plt.xlim([0, 100])
        plt.ylim([0, 0.4])
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
        #plt.show(block=False)
        li.set_ydata(y)
        fig.canvas.draw()
        time.sleep(0.01)

        if ear < LIMIAR_EAR:
            CONTADOR += 1
            if CONTADOR >= QTD_CONSEC_FRAMES:
                if not ALARME_ON:
                    ALARME_ON = True
                    if alarme:
                        #t = Thread(target=tocar_alarme, args=("alarm.wav",))
                        #t.deamon = True
                        #t.start()
                        tocar_alarme("alarm.wav")

                cv2.putText(frame, "[ALERTA] SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            CONTADOR = 0
            ALARME_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Atualize o vídeo e o gráfico a cada TAXA_ATUALIZACAO frames
        if contador_frames % TAXA_ATUALIZACAO == 0:
            frame_slot.image(frame, channels="BGR")
            graph_slot.pyplot(fig)

    frame_slot.image(frame, channels="BGR")
    graph_slot.pyplot(fig)
    # ... [restante do código dentro do loop]

    # Atualize o contador de frames
    contador_frames += 1

vs.stop()

