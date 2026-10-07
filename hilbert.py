# %%
import numpy as np

#test 1D
def custom_hilbert(x):
    """
    Param:
    -------
    x: Tín hiệu đầu vào (mảng 1D)

    Return:
    -------
        z : ndarray (complex) -> Analytic Signal z(t)
        h : ndarray (real)    -> Biến đổi Hilbert h(t) (Phần ảo của z)
        IA_1D : ndarray (real)   -> Biên độ tức thời Instantaneous Amplitude (IA)
        IP_1D : ndarray (real)   -> Pha tức thời Instantaneous Phase (IP)
        IF_1D : ndarray (real)-> Tần số tức thời Instantaneous Frequency (IF)
    """
    N = len(x)

    #1. Chuyển sang miền tần số bằng FFT
    X = np.fft.fft(x)

    #2. Khởi tạo vector đáp ứng tần số h(f)
    H = np.zeros(N)

    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1 : N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1 : (N + 1) // 2] = 2.0

    #3. Nhân trong miền tần số và tính IFFT để thu được Analytic Signal z(t)
    Z_1D = X * H
    z_1D = np.fft.ifft(Z_1D)


    #4. Trích xuất h, IA, IP
    h_1D = np.imag(z_1D)
    IA_1D = np.abs(z_1D)  
    IP_1D = np.angle(z_1D)  

    # Bước 5: Tính IF từ đạo hàm của IP 
    IP_unwrapped = np.unwrap(IP_1D)
    IF_1D = np.diff(IP_unwrapped) / (2.0 * np.pi)
    # Thêm 1 phần tử ở cuối để giữ nguyên độ dài mảng ban đầu
    IF_1D = np.append(IF_1D, IF_1D[-1])

    return z_1D, h_1D, IA_1D, IP_1D, IF_1D

# %%
#multi-D
def custom_hilbert_2d(imfs, axis=-1):
    """Tính HHT và các tham số tức thời cho mảng 2D/multi-D.
    Param:
    --------
    imfs : ndarray 2D
        Mảng tín hiệu/các IMF đầu vào (shape (n_imfs, n_samples)).
    axis : int, optional
        Chiều thời gian để thực hiện phép biến đổi (default axis=-1).
    
    Returns:
    --------
    z : ndarray (complex) -> Analytic Signal z(t)
    h : ndarray (real)    -> Biến đổi Hilbert h(t) (Phần ảo của z)
    IA : ndarray (real)   -> Biên độ tức thời Instantaneous Amplitude (IA)
    IP : ndarray (real)   -> Pha tức thời Instantaneous Phase (IP)
    IF : ndarray (real)-> Tần số tức thời Instantaneous Frequency (IF)
    """
    imfs = np.asarray(imfs)
    N = imfs.shape[axis]  # Số mẫu tín hiệu theo chiều thời gian

    # 1. Chuyển sang miền tần số bằng FFT theo axis
    X = np.fft.fft(imfs, axis=axis)

    # 2. Xây dựng vector đáp ứng tần số h(f) độ dài N
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1 : N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1 : (N + 1) // 2] = 2.0

    shape_H = [1] * imfs.ndim
    shape_H[axis] = N
    H = H.reshape(shape_H)

#3. Nhân trong miền tần số và tính IFFT để thu được Analytic Signal z(t)
    Z = X * H
    z = np.fft.ifft(z, axis=axis)

    #4. Trích xuất h, IA, IP
    h = np.imag(z)  
    IA = np.abs(z)  
    IP = np.angle(z)  

    #5. Tính IF từ đạo hàm của IP
    IP_unwrapped = np.unwrap(IP, axis=axis)

    IF = np.diff(IP_unwrapped, axis=axis) / (2.0 * np.pi)

    last_col = np.take(IF, [-1], axis=axis)
    IF = np.concatenate([IF, last_col], axis=axis)

    return z, h, IA, IP, IF



