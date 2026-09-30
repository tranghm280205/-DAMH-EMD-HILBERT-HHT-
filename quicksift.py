# %%
import matplotlib.pyplot as plt
import numpy as np
import emd

# %%
#mô phỏng sóng phi tuyến ở tần số 5Hz và sóng hình sin ở tần số 1Hz
sample_rate = 1000  #tần số lấy mẫu
seconds = 10  # độ dài tín hiệu
num_samples = sample_rate*seconds

t = np.linspace(0, seconds, num_samples)  #vector thời gian rời rạc

a = 5      # Tần số sóng phi tuyến (Hz)
b = 2.2e-1 # Tần số sóng hình sin (Hz)
c = 1      # Tần số sóng hình cos (Hz)

#Thay đổi mức độ biến dạng dạng sóng sin [-1 đến 1]
nonlinearity_deg = 0.25

# Thay đổi độ lệch trái-phải (pha lệch) của biến dạng [-pi tới pi]
nonlinearity_phi = -np.pi/4

# Sóng hình sin chuẩn ở tần số b Hz
sin_wave = np.sin(2 * np.pi * b * t)

# Sóng hình cos chuẩn ở tần số c Hz
cos_wave = np.cos(2 * np.pi * c * t)

# Sóng phi tuyến ở tần số a Hz
# Công thức: 𝑦(𝑡)=sin⁡(𝜔_𝑎⋅𝑡)+𝛼⋅sin⁡(2𝜔_𝑎⋅𝑡+𝜑)
omega_a = 2 * np.pi * a * t
nonlinear_wave1 = np.sin(omega_a) + nonlinearity_deg * np.sin(2 * omega_a + nonlinearity_phi)

# Tổng hợp hai sóng
signal_nonlinear = nonlinear_wave1 + cos_wave - sin_wave


# %%
plt.figure(figsize=(12, 16))

# Đồ thị Sóng sin b Hz
plt.subplot(4, 1, 1)
plt.plot(sin_wave, color='blue')
plt.title(f'Sóng sin (Tần số b = {b} Hz)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

# Đồ thị Sóng cos c Hz
plt.subplot(4, 1, 2)
plt.plot(cos_wave, color='blue')
plt.title(f'Sóng cos (Tần số c = {c} Hz)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

# Đồ thị Sóng phi tuyến a Hz
plt.subplot(4, 1, 3)
plt.plot(nonlinear_wave1, color='red')
plt.title(f'Sóng phi tuyến (a = {a} Hz | deg = {nonlinearity_deg}, phi = -pi/4 rad)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

# Đồ thị Sóng tổng hợp (phổ thông)
plt.subplot(4, 1, 4)
plt.plot(signal_nonlinear, color='purple')
plt.title('Tín hiệu tổng hợp (phổ thông)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

plt.tight_layout()
plt.show()

# %%
#mô phỏng sóng + nhiễu 
mean = 0
std_dev = 0.2
size = len(t)
noise = np.random.normal(mean, std_dev, size)

signal_noise = signal_nonlinear + noise
print(noise)

# %%
# Đồ thị Sóng nhiễu
plt.plot(signal_nonlinear, color = 'green', linewidth=0.5)
plt.plot(signal_noise, color='red', linewidth=0.05)
plt.title('Tín hiệu phi tuyến vs có nhiễu')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

plt.tight_layout()
plt.show()

# %%
# Mô phỏng tín hiệu biến thiên theo thời gian
from scipy.signal import chirp

signal_chirp = chirp(
    t,
    f0=5,
    f1=100,
    t1=t[-1],
    method='linear'
)
signal_chirp

# %%
# Đồ thị Tín hiệu biến thiên
plt.figure(figsize=(16, 8))
plt.plot(signal_chirp, color = 'purple', linewidth=0.5)
plt.title('Tín hiệu biến thiên theo thời gian')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

plt.tight_layout()
plt.show()

# %%
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema


def extract_single_imf(signal, sd_threshold=0.2, max_iterations=1000):
    """
    Trích xuất 1 thành phần IMF duy nhất từ tín hiệu truyền vào.
    sd_threshold: Ngưỡng quyết định sự hội tụ của sifting.
    max_iterations: Quyết định số vòng lặp tối đa
    """ 
    h = signal.copy()
    t = np.arange(len(signal))

    for iteration in range(max_iterations):
        # 1. Tìm các chỉ số cực đại và cực tiểu cục bộ
        max_idx = argrelextrema(h, np.greater)[0]
        min_idx = argrelextrema(h, np.less)[0]

        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        # 2. Xử lý biên bằng reflect padding
        max_idx_ext = np.r_[2*max_idx[0] - max_idx[1:2], max_idx, 2*max_idx[-1] - max_idx[-2:-1]]
        max_val_ext = np.r_[h[max_idx[1:2]], h[max_idx], h[max_idx[-2:-1]]]
        
        min_idx_ext = np.r_[2*min_idx[0] - min_idx[1:2], min_idx, 2*min_idx[-1] - min_idx[-2:-1]]
        min_val_ext = np.r_[h[min_idx[1:2]], h[min_idx], h[min_idx[-2:-1]]]

        max_idx_ext = np.clip(max_idx_ext, 0, len(h) - 1)
        min_idx_ext = np.clip(min_idx_ext, 0, len(h) - 1)

        # 3. Nội suy đường bao bằng CubicSpline
        cs_max = CubicSpline(max_idx_ext, h[max_idx_ext], bc_type="natural")
        cs_min = CubicSpline(min_idx_ext, h[min_idx_ext], bc_type="natural")

        e_max = cs_max(t)
        e_min = cs_min(t)

        # 4. Tính đường trung bình
        m = (e_max + e_min) / 2.0

        # 5. Trừ đường trung bình ra khỏi tín hiệu
        h_next = h - m

        # 6. Tính tiêu chuẩn dừng SD 
        sd = np.sum((h - h_next) ** 2) / (np.sum(h**2) + 1e-10)

        h = h_next

        if sd < sd_threshold:
            break

    return h


def custom_sift(signal, max_imfs=5):
    """Phân tách tín hiệu thành mảng danh sách các IMF và Residual."""
    imfs = []
    res = signal.copy()

    for _ in range(max_imfs):
        imf = extract_single_imf(res)

        if np.allclose(imf, 0) or len(argrelextrema(imf, np.greater)[0]) < 2:
            break

        imfs.append(imf)

        res = res - imf

    return np.array(imfs), res        


# %%
imf_nonlinear_manualEDM, residual_nonlinear_manualEMD = custom_sift(signal_nonlinear)
imf_nonlinear_pyEDM = emd.sift.sift(signal_nonlinear).T

imf_noise_manualEDM, residual_noise_manualEMD = custom_sift(signal_noise)
imf_noise_pyEDM = emd.sift.sift(signal_noise).T

imf_chirp_manualEDM, residual_chirp_manualEMD = custom_sift(signal_chirp)
imf_chirp_pyEDM = emd.sift.sift(signal_chirp).T

# %%
reconstructed_nonlinear_manual = np.sum(imf_nonlinear_manualEDM, axis=0) + residual_nonlinear_manualEMD

plt.plot(signal_nonlinear, color = 'green', linewidth=0.5)
plt.plot(reconstructed_nonlinear_manual, color='red', linewidth=0.3)
plt.title('Tín hiệu phi tuyến gốc vs tái tạo')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.xlim(0, num_samples)

plt.tight_layout()
plt.show()


