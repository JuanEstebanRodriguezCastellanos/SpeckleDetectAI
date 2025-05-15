import numpy as np
from PIL import Image, ImageFilter
import cv2

class Filtro:
    def __init__(self):
        self.L = 1
        self.iteraciones = 2
        self.porcentaje_borde = 0.6

    def aplicar(self, imagen_pil):
        img = imagen_pil.convert('L')  # Convertir a escala de grises
        ip = np.array(img, dtype=np.float32)
        
        for _ in range(self.iteraciones):
            # Otsu
            otsu_val = self.otsu_threshold(ip)
            otsu_ip = (ip > otsu_val).astype(np.uint8)

            temp = np.zeros_like(ip)

            for i in range(ip.shape[0]):
                for j in range(ip.shape[1]):
                    idx = i * ip.shape[1] + j
                    if otsu_ip[i, j] == 0:
                        temp[i, j] = self.calcular_valor(ip, i, j, self.L * 3)
                    else:
                        temp[i, j] = self.calcular_valor(ip, i, j, self.L * 0.3)
            
            ip = temp
            ip = self.mejorar_bordes(ip, otsu_ip)

        return Image.fromarray(np.clip(ip, 0, 255).astype(np.uint8))

    def otsu_threshold(self, img):
        hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
        total = img.size
        sum_total = np.dot(np.arange(256), hist.ravel())
        sum_b, w_b, var_max, threshold = 0.0, 0.0, 0.0, 0

        for i in range(256):
            w_b += hist[i]
            if w_b == 0: continue
            w_f = total - w_b
            if w_f == 0: break
            sum_b += i * hist[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold = i
        return threshold

    def calcular_valor(self, ip, y_centro, x_centro, localL):
        height, width = ip.shape
        distancias = [1, 2, 3, 4, 5]
        pos = 0

        while pos < 4:
            psi_wn = self.calcular_flusser(ip, x_centro, y_centro, distancias[pos])
            psi_bne = self.calcular_flusser(ip, x_centro, y_centro, distancias[pos + 1])
            if abs(psi_wn - psi_bne) <= 0.1:
                pos += 1
            else:
                break

        distancia = distancias[pos]
        x_min = max(0, x_centro - distancia)
        x_max = min(width, x_centro + distancia + 1)
        y_min = max(0, y_centro - distancia)
        y_max = min(height, y_centro + distancia + 1)

        vecindad = ip[y_min:y_max, x_min:x_max]
        media = np.mean(vecindad)
        varianza = np.var(vecindad, ddof=1)

        varianza_nl = max(0, (localL * varianza - media) / (localL + 1))
        valor = media + (varianza_nl * (ip[y_centro, x_centro] - media)) / (varianza_nl + (media * media + varianza_nl) / self.L)

        return min(255, max(0, valor))

    def mejorar_bordes(self, ip, otsu):
        borroso = cv2.GaussianBlur(ip, (3, 3), 1)
        resultado = np.copy(ip)

        mask = otsu > 0
        mejorado = ip[mask] + (ip[mask] - borroso[mask]) * self.porcentaje_borde
        resultado[mask] = np.clip(mejorado, 0, 255)
        return resultado
    
    def calcular_flusser(self, ip, x_centro, y_centro, dis):
        height, width = ip.shape

        x_min = max(0, x_centro - dis)
        x_max = min(width, x_centro + dis + 1)
        y_min = max(0, y_centro - dis)
        y_max = min(height, y_centro + dis + 1)

        vecindad = ip[y_min:y_max, x_min:x_max]
        moments = cv2.moments(vecindad)

        psi1 = moments['mu20'] + moments['mu02']
        psi2 = (moments['mu30'] + moments['mu12']) ** 2 + (moments['mu21'] + moments['mu03']) ** 2
        psi3 = (moments['mu20'] - moments['mu02']) * ((moments['mu30'] + moments['mu12']) ** 2 - (moments['mu21'] + moments['mu03']) ** 2)
        psi4 = moments['mu11'] * ((moments['mu30'] + moments['mu12']) ** 2 + (moments['mu03'] + moments['mu21']) ** 2)
        psi5 = (moments['mu30'] - 3 * moments['mu12']) * (moments['mu30'] + moments['mu12'])
        psi6 = (3 * moments['mu21'] - moments['mu03']) * (moments['mu30'] + moments['mu12'])

        psi_sum = sum(np.sign(p) * np.log(abs(p)) if abs(p) > 1e-10 else 0 for p in [psi1, psi2, psi3, psi4, psi5, psi6])

        return psi_sum
