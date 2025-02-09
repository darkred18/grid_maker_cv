import cv2 
import io, os.path, copy
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

R, G, B = (0,0,255), (0,255,0), (255,0,0)
width_center = 0
height_center = 0

def init_state():
    print('init_state')
    if 'thickness' not in st.session_state:
        st.session_state.thickness= 2
    if 'line_nums' not in st.session_state:
        st.session_state.line_nums= 1

    if 'width_offset' not in st.session_state:
        st.session_state.width_offset = 0
    if 'height_offset' not in st.session_state:
        st.session_state.height_offset = 0

    if 'center_color' not in st.session_state:
        st.session_state.center_color = '#20BFCE'
    if 'selected_color' not in st.session_state:
        st.session_state.selected_color = '#E64A31'

    if 'shape_size' not in st.session_state:
        st.session_state.shape_size = 0

    if 'radio_shape' not in st.session_state:
        st.session_state.radio_shape = 'circle'

    if 'color_check' not in st.session_state:
        st.session_state.color_check = False

    if 'color_nums' not in st.session_state:
        st.session_state.color_nums = 5

def get_bufImage(_image):
    buffer = io.BytesIO()
    _image.save(buffer, format='PNG')
    return buffer

def get_cvBufImage(_image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 품질을 90으로 설정 (0-100)
    _, buffer = cv2.imencode('.jpg', _image, encode_param)
    byte_im = buffer.tobytes()
    return byte_im

def convert_opencv_image(file):
    # file = uploaded_file.read()
    file_bytes = np.asarray(bytearray(file), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return img_rgb

def setup_sidebar():
    print('setup_sidebar')
    ex_file = st.sidebar.expander('LOAD IMAGE',expanded=True)
    ex_color = st.sidebar.expander('LINE COLOR')
    ex_line = st.sidebar.expander('LINE OPTION')
    ex_offset = st.sidebar.expander('LINE OFFSET')
    ex_circle = st.sidebar.expander('GUIDE OPTION')
    # ex_t_color= st.sidebar.expander('COLOR THEME')
    

    load_file = ex_file.file_uploader('이미지를 업로드 하세요.')   #, type=['png', 'jpg', 'jpeg'])

    if not load_file:
        return (None,None,None)
    
    # name , ext = load_file.name.split('.')
    name , ext = os.path.splitext(load_file.name)
    _image = convert_opencv_image(load_file.read())

    h,w,_ = _image.shape
    w_offset = int(w * 0.5)
    h_offset = int(h * 0.5)

    min_size = w_offset if h_offset > w_offset else h_offset
  
    columns = ex_color.columns([1,1])
    with columns[0]:
        st.session_state.center_color = st.color_picker('center color', st.session_state.center_color )
    with columns[1]:
        st.session_state.selected_color = st.color_picker("색상을 선택하세요", st.session_state.selected_color)    

    st.session_state.line_nums = ex_line.slider("라인 갯수는?", 0, 10, (st.session_state.line_nums))
    st.session_state.thickness = ex_line.slider('두께',1,10,(st.session_state.thickness))
    st.session_state.width_offset = ex_offset.slider("가로 offset", -w_offset, w_offset, (0))
    st.session_state.height_offset = ex_offset.slider("세로 offset", -h_offset, h_offset, (0))

    st.session_state.radio_shape = ex_circle.radio(label='SHAPE TYPE', options=['circle','rect'], horizontal=True)

    st.session_state.shape_size = ex_circle.slider('Shape Size',0, min_size,(0))
    st.session_state.color_check = st.sidebar.checkbox('색상 팔레트')
    if st.session_state.color_check:
        st.session_state.color_nums = st.sidebar.slider('팔레트 갯수',1,10,(3))
    # if radio == 'circle':
    #     print('c')
        
    # elif radio == 'rect':
    #     print('r')

    return (_image, name, ext)

def grid_line(_image):
    print('grid_line')
    global width_center, height_center
    # _image, _name, _ext = img_info

    line_nums = st.session_state.line_nums
    thickness = st.session_state.thickness
    h,w ,_ = _image.shape

    center_color = st.session_state.center_color
    selected_color = st.session_state.selected_color

    if line_nums <= 0:
        return st.image(_image)
    
    line_margin = w // line_nums
        
    width_center = w // 2  + st.session_state.width_offset

    height_center = h // 2 + st.session_state.height_offset

    cv2.line(_image, (width_center,0), (width_center,h), G, st.session_state.thickness)
    cv2.line(_image, (0,height_center), (w,height_center), G, st.session_state.thickness)


    for i in range(1,line_nums):
        cv2.line(_image, (line_margin*i+width_center,0), (line_margin*i+width_center,h), R, st.session_state.thickness)
        cv2.line(_image, (0,line_margin*i+height_center), (w,line_margin*i+height_center), R, st.session_state.thickness)
 
        cv2.line(_image, (width_center-line_margin*i,0), (width_center-line_margin*i,h), R, st.session_state.thickness)
        cv2.line(_image, (0,height_center-line_margin*i), (w,height_center-line_margin*i), R, st.session_state.thickness)

    # buffer_img = get_bufImage(_image)

    # st.sidebar.download_button(
    #     label = "이미지 다운로드",
    #     data = buffer_img,
    #     file_name = _name + '_grid.' + _ext,
    #     mime = "image/%s"%_ext)

    # st.image(_image, channels='BGR')
    # _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    return _image

def save_image(_image, _name, _ext):
    buffer_img = get_cvBufImage(_image)
    st.sidebar.download_button(
        label = "이미지 다운로드",
        data = buffer_img,
        file_name = _name + '_grid' + _ext,
        mime = "image/%s"%_ext)


def guide_shape(_image):
    s_size = st.session_state.shape_size
    if st.session_state.radio_shape == 'circle':
        cv2.circle(_image, (width_center,height_center), s_size,B,st.session_state.thickness)
    elif st.session_state.radio_shape == 'rect':
        x = width_center  - s_size
        y = height_center - s_size
        w = width_center  + s_size
        h = height_center + s_size
        cv2.rectangle(_image,(x,y),(w,h),B,st.session_state.thickness)
    return _image

def get_mean_color(_image):
    # 이미지를 2차원 배열로 변환
    pixels = _image.reshape(-1, 3)

    # K-means 클러스터링을 사용하여 주요 색상 추출
    kmeans = KMeans(n_clusters=st.session_state.color_nums)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # 각 클러스터의 픽셀 수 계산
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # 색상과 픽셀 수를 결합하여 정렬
    sorted_colors = sorted(zip(colors, counts), key=lambda x: x[1], reverse=True)
    sorted_colors = [color for color, count in sorted_colors]

    # 주요 색상 시각화
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    for idx, color in enumerate(sorted_colors):
        ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=color/255))
    ax.set_xlim(0, len(sorted_colors))
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Streamlit 앱
    st.title('주요 색상 추출 예제')
    st.pyplot(fig)

if __name__ == '__main__':
    init_state()
    org_img, name, ext = setup_sidebar()
    img = copy.deepcopy(org_img)
    if org_img is not None:
        img = grid_line(img)
        img = guide_shape(img)
        save_image(img,name,ext)
        st.image(img)
        # st.image(org_img)
    if img is not None and st.session_state.color_check:
        get_mean_color(org_img)
