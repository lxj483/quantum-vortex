import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import vortex

class VortexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vortex 检测")
        
        # 参数默认值
        self.params = {
            'min_radius': tk.IntVar(value=20),  # 改为IntVar
            'max_radius': tk.IntVar(value=50), # 改为IntVar
            'color_threshold': tk.DoubleVar(value=0.9),
            'split': tk.DoubleVar(value=0.6),
            'more_precise': tk.IntVar(value=3),
            'inverse' : tk.BooleanVar(value=False)  
        }
        
        # 创建控件
        self.create_widgets()
        
        # 存储变量
        self.image_path = None
        self.tk_image = None
        self.vortices = []
        self.original_image = None

    def create_widgets(self):
        # 控制面板框架
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 文件操作按钮
        self.btn_open = tk.Button(control_frame, text="打开图片", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # 参数调节控件
        param_frame = tk.LabelFrame(control_frame, text="检测参数")
        param_frame.pack(side=tk.LEFT, padx=10)
        
        self.sliders = {}
        row = 0
        
        # # 阈值滑块
        # tk.Label(param_frame, text='threshold').grid(row=row, column=0, sticky='e')
        # slider = ttk.Scale(param_frame, from_=0, to=1, value=self.params['threshold'],
        #                   command=lambda v: self.update_param('threshold', float(v)))
        # slider.grid(row=row, column=1, padx=5, pady=2)
        # self.sliders['threshold'] = slider
        # row += 1
        
        # min_radius 数值输入
        for param in['min_radius','max_radius','more_precise']:
            tk.Label(param_frame, text=param).grid(row=row, column=0, sticky='e')
            spinbox = tk.Spinbox(param_frame, from_=1, to=50, textvariable=self.params[param],
                                command=lambda: self.update_param(param, self.params[param]))
            spinbox.grid(row=row, column=1, padx=5, pady=2)
            row += 1
        

        
        # 其他滑块参数
        for param in ['color_threshold', 'split']:
            tk.Label(param_frame, text=param).grid(row=row, column=0, sticky='e')
            slider = ttk.Spinbox(param_frame, from_=0, to=1, textvariable=self.params[param],
                              command=lambda: self.update_param(param, self.params[param]),increment=0.1)
            slider.grid(row=row, column=1, padx=5, pady=2)
            self.sliders[param] = slider
            row += 1
        
        # more_precise 复选框
        check = tk.Checkbutton(param_frame, text="inverse", 
                              variable=self.params['inverse'],
                              command=lambda: self.update_param('inverse', self.params['inverse']))
        check.grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1
        # 检测按钮
        self.btn_detect = tk.Button(control_frame, text="检测涡旋", command=self.detect_vortices)
        self.btn_detect.pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 图像显示Canvas
        self.canvas = tk.Canvas(result_frame, width=600, height=400, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 坐标显示文本框
        self.text_result = tk.Text(result_frame, width=30, height=10)
        self.text_result.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    def update_param(self, param, value):
        self.params[param] = value
        if self.image_path:  # 如果有图片则自动重新检测
            self.detect_vortices()
    
    def open_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=(("图片文件", "*.png *.jpg *.jpeg"), ("所有文件", "*.*"))
        )
        if filepath:
            self.image_path = filepath
            self.original_image = cv2.imread(filepath)
            self.show_image(self.original_image)
    
    def show_image(self, image):
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL格式
        pil_image = Image.fromarray(image)
        
        # 调整大小以适应Canvas
        w, h = pil_image.size
        ratio = min(600/w, 400/h)
        new_size = (int(w*ratio), int(h*ratio))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        
        # 转换为Tkinter格式
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # 显示图片
        self.canvas.delete("all")
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # 显示坐标点
        for (x, y) in self.vortices:
            x_scaled = x * ratio
            y_scaled = y * ratio
            self.canvas.create_oval(x_scaled-3, y_scaled-3, x_scaled+3, y_scaled+3, 
                                  outline='red', width=2)
    
    def detect_vortices(self):
        if self.image_path:
            # 调用vortex检测函数
            self.vortices = vortex.detect_vortices_by_convolution(
                self.image_path,
                min_radius=self.params['min_radius'].get(),
                max_radius=self.params['max_radius'].get(),
                color_threshold=self.params['color_threshold'].get(),
                split=self.params['split'].get(),
                more_precise=self.params['more_precise'].get(),
                inverse=self.params['inverse'].get()
            )
            
            # 显示结果
            self.show_image(self.original_image)
            
            # 在文本框中显示坐标
            self.text_result.delete(1.0, tk.END)
            for i, (x, y) in enumerate(self.vortices):
                self.text_result.insert(tk.END, f"涡旋 {i+1}: ({x:.1f}, {y:.1f})\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = VortexGUI(root)
    root.mainloop()
