import cv2
import numpy as np
import logging

class OpticalFlowProcessor:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.prev_frame = None

        # پارامترهای Optical Flow
        self.vertical_motion_threshold = config.opticalflow.VERTICAL_MOTION_THRESHOLD
        self.glowing_pixels = config.opticalflow.GLOWING_PIXELS
        self.resize_width = config.opticalflow.RESIZE_WIDTH
        self.resize_height = config.opticalflow.RESIZE_HEIGHT
        self.x_min_min = config.opticalflow.X_MIN_MIN
        self.x_max_max = config.opticalflow.X_MAX_MAX
        self.min_active_columns = config.opticalflow.MIN_ACTIVE_COLUMNS
        self.min_active_rows = config.opticalflow.MIN_ACTIVE_ROWS
        self.strip_padding = config.opticalflow.STRIP_PADDING

        # متغیرهای حالت برای نگهداری کراپ قبلی
        self.y_min_old = 0
        self.y_max_old = 0
        self.x_min_old = 0
        self.x_max_old = 0
        self.moving = False

    def detect_and_crop(self, current_frame):
        """
        تشخیص میله و کراپ تصویر در یک متد واحد
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False, current_frame

        try:
            # تغییر اندازه تصاویر
            img1 = cv2.resize(
                self.prev_frame, (self.resize_width, self.resize_height))
            img2 = cv2.resize(
                current_frame, (self.resize_width, self.resize_height))

            # تبدیل به خاکستری
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # حذف پیکسل‌های روشن (ناشی از گلو)
            mask = ((gray1 >= self.glowing_pixels) &
                    (gray2 >= self.glowing_pixels))

            # بهبود کنتراست
            equalized1 = cv2.equalizeHist(gray1)
            equalized2 = cv2.equalizeHist(gray2)
            equalized1[mask] = 0
            equalized2[mask] = 0

            # محاسبه Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                equalized1, equalized2, None,
                0.3, 5, 21, 3, 7, 1.5, 0
            )

            # استخراج حرکت عمودی
            dy = flow[..., 1]
            motion_mask = np.abs(dy) > self.vertical_motion_threshold

            # محدود کردن به ناحیه مورد نظر
            limited_motion_mask = motion_mask[:, self.x_min_min:self.x_max_max]

            # بررسی فعالیت در ستون‌ها
            col_activity = np.sum(limited_motion_mask, axis=0)
            active_cols = np.where(col_activity > self.resize_height * 0.1)[0]
            active_cols = active_cols + self.x_min_min

            # تشخیص وجود میله
            has_rod = len(active_cols) >= self.min_active_columns

            # کراپ تصویر در صورت وجود حرکت
            cropped_frame = current_frame
            if has_rod:
                x_min = max(int(np.min(active_cols)) +
                            self.strip_padding, self.x_min_min)
                x_max = min(int(np.max(active_cols)) -
                            self.strip_padding, self.x_max_max)

                # استخراج ماسک حرکت در نوار فعال
                motion_strip = motion_mask[:, x_min:x_max]

                # بررسی فعالیت در سطرها
                row_activity = np.sum(motion_strip, axis=1)
                active_rows = np.where(row_activity > (x_max - x_min) * 0.2)[0]

                if len(active_rows) >= self.min_active_rows:
                    y_min = max(int(np.min(active_rows)), 0)
                    y_max = min(int(np.max(active_rows)), self.resize_height)

                    if self.moving:
                        if y_min == 0:
                            cropped_frame = current_frame[
                                self.y_min_old:self.y_max_old,
                                self.x_min_old:self.x_max_old
                            ]
                        else:
                            cropped_frame = current_frame[y_min:y_max,
                                                          x_min:x_max]

                    # ذخیره مختصات فعلی
                    self.y_min_old = y_min
                    self.y_max_old = y_max
                    self.x_min_old = x_min
                    self.x_max_old = x_max
                    self.moving = True

            # به‌روزرسانی فریم قبلی
            self.prev_frame = current_frame.copy()
            return has_rod, cropped_frame

        except Exception as e:
            self.logger.error(f"Optical Flow error: {e}")
            return False, current_frame

    def reset(self):
        """ریست متغیرهای حالت"""
        self.prev_frame = None
        self.y_min_old = 0
        self.y_max_old = 0
        self.x_min_old = 0
        self.x_max_old = 0
        self.moving = False


