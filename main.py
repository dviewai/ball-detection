import tensorflow as tf
import sys
import cv2
import numpy as np
sys.path.append("..")
import time
import os
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.core.image import Image
from kivy.uix.widget import Widget
from kivy.base import EventLoop
from kivy.config import Config
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.uix.dropdown import DropDown
from kivy.core.window import Window
import csv
import collections

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=2,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):

  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
            print(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]


  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (int(xmin * 600), int(xmax * 600), int(ymin * 600), int(ymax * 600))
    image = cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 1)

    return image, (int(left), int(right), int(top), int(bottom)),display_str
  return image, [], 0

f = n = height_px = width_px = toss_speed_px = graph_count = serve_speed_px = graph_update = flag_value = peak_over = plot_update = 0
cap_str=''
x=[]
y=[]
sc =[]
top = []
safe_arr = []
N=[]
cap=cv2.VideoCapture(' ')
img = cv2.imread('gui_images/img.jpg')


def moving_forward(y):
    for i in range (1, 3):
      if y[i-1]-y[i]>5:
        return False
    return True

def moving_reverse(y):
    for i in range (1, 3):
      if y[i-1]-y[i]<5:
        return False
    return True

def returning_back(y):
    for i in range(1, len(y)-1):
      if y[i]>=y[i-1]:
        if y[i]-y[i+1]>5:
          return True
      else:
        if y[i+1]-y[i]>5:
          return True
    return False

def peak_sequence(x, y, N, sc, flag):
    up=down=0
    flag_up=flag_down=0
    cleared_x = [x[0]]
    cleared_y = [y[0]]
    cleared_N = [N[0]]
    cleared_sc = [sc[0]]

    if flag:
      for i in range(1, len(x)):  
        if flag_down == 0 and (x[i]-x[i-1])<2 and (y[i-1]-y[i])<2:
          up += 1
          flag_up = 1
          cleared_x.append(x[i])
          cleared_y.append(y[i])
          cleared_N.append(N[i])
          cleared_sc.append(sc[i])

        elif flag_up == 1 and (x[i-1]-x[i])<2 and (y[i-1]-y[i])<2:
          down+=1
          flag_down = 1
          cleared_x.append(x[i])
          cleared_y.append(y[i])
          cleared_N.append(N[i])
          cleared_sc.append(sc[i])
        if flag_down and flag_up and up<10:
            cleared_x = []
            cleared_y = []
            up=0
            flag_up = 0
      if len(y)>3:
        if returning_back(y[-3:]):
          cleared_x=[]
          cleared_y=[]
          cleared_sc=[]
          cleared_N=[]
      
      if flag_down and flag_up and when_does_the_ball_got_hit([x[-2],x[-1]],[y[-2],y[-1]],[N[-2],N[-1]]):
        # print ([x[-2],x[-1]],[y[-2],y[-1]],[N[-2],N[-1]])
        return cleared_x, cleared_y, cleared_N, cleared_sc,1
      else:
        return cleared_x, cleared_y, cleared_N, cleared_sc,0

    else:
      for i in range(1, len(x)):  
        if flag_down == 0 and (x[i]-x[i-1])<2  and (y[i]-y[i-1])<2:
          up += 1
          flag_up = 1
          cleared_x.append(x[i])
          cleared_y.append(y[i])
          cleared_N.append(N[i])
          cleared_sc.append(sc[i])

        elif  flag_up == 1 and (x[i-1]-x[i])<2 and (y[i]-y[i-1])<2:
          down+=1
          flag_down = 1
          cleared_x.append(x[i])
          cleared_y.append(y[i])
          cleared_N.append(N[i])
          cleared_sc.append(sc[i])

          if flag_down and flag_up and up<10:
            cleared_x = []
            cleared_y = []
            up=0
            flag_up = 0

      if flag_down and flag_up and when_does_the_ball_got_hit([x[-2],x[-1]],[y[-2],y[-1]],[N[-2],N[-1]]):
        return cleared_x, cleared_y, cleared_N, cleared_sc,1
      else:
        return cleared_x, cleared_y, cleared_N, cleared_sc,0

def starting_is_it_moving_up(x):
    f=0
    for i in range(0,2):
      if x[i]-x[i+1]>0:
        f=1
    if f==0: 
      return False
    return True

def speed_at_each_step(x,y,n):
    return (((x[0]-y[0])**2 + (x[1]-y[1])**2 )**(0.5))/n

def when_does_the_ball_got_hit(x,y,n):
    global f
    if y[1]-y[0]>=10 and n[1]-n[0]<=2:
      return 1
    return 0

def clear_previous_peaks(x,y,n):
    return [], [], []

def ball_goes_out(x,y,side):
    if side:
      if max(y)>=400:
        return True
      return False
    else:
      if min(y)<=100:
        return True
      return False

def start_recording():
  global height_px, width_px, toss_speed_px, serve_speed_px, x, y, N, sc, peak_over, flag_value
  if len(x)>=3: 
    
    if moving_forward(y):
      if not(peak_over):
        x, y, N, sc, peak_over = peak_sequence(x, y, N, sc, 1)
      if len(x)>=3 and peak_over:
        if flag_value==0:
          toss_speed_px = (sum(x[:3]))/(((N[1]-(N[1]-1))+(N[2]-(N[1]-1))+(N[3]-(N[1]-1))))
          init_x = x[0]
          init_y = y[0]
          peak_x = min(x)
          peak_y = y[x.index(peak_x)]
          hit_x = x[-2]
          hit_y = y[-2]
          height_px = init_x - peak_x 
          width_px = 2*(hit_y-peak_y)
          serve_speed_px = ((y[-1]-y[-2]))/((N[-1]-(N[-1]-1))-(N[-2]-(N[-1]-1)))
          flag_value = 1
        if ball_goes_out(x,y,1):
          peak_over = 0
          return height_px, width_px, toss_speed_px, serve_speed_px, 1
        elif returning_back(y[-3:]):
          print 'ss' 
          x = []
          y = []
          sc = []
          N = []

          return height_px, width_px, toss_speed_px, serve_speed_px, 0
        else:
          print 'sabaa'
          


    elif moving_reverse(y):
      if not(peak_over):
        x, y, N, sc, peak_over = peak_sequence(x, y, N, sc, 1)
      if len(x)>=3 and peak_over:
        if flag_value==0:
          toss_speed_px = (sum(x[:3]))/(((N[1]-(N[1]-1))+(N[2]-(N[1]-1))+(N[3]-(N[1]-1))))
          init_x = x[0]
          init_y = y[0]
          peak_x = min(x)
          peak_y = y[x.index(peak_x)]
          hit_x = x[-2]
          hit_y = y[-2]
          height_px = abs(init_x - peak_x)
          width_px = abs(2*(hit_y-peak_y))
          serve_speed_px = abs(((y[-1]-y[-2]))/(N[-1]-N[-2]))
          flag_value = 1
        if ball_goes_out(x,y,0):
          peak_over = 0
          return height_px, width_px, toss_speed_px, serve_speed_px, 1

    else:
      x = []
      y = []
      sc = []
      N = []
  return height_px, width_px, toss_speed_px, serve_speed_px, 0


def storage_for_file_name(t):
  global x,y,N,cap
  x=[]
  y=[]
  N =[]
  if t != '0':
    cap_str='data/'+str(t)+'.mp4'
    cap = cv2.VideoCapture(cap_str)
  elif t=='stop':
    cap = cv2.VideoCapture(1)
  else:
    cap = cv2.VideoCapture(0)
  


def save_plot(x):
  global plot_update
  print (x[5:])
  plot_update = int(x[5:])
  



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('inference_graph/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print('imported everything and graph is loaded')


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    class LoginScreen(FloatLayout):
      str1 = StringProperty()
      str2 = StringProperty()
      img_src = StringProperty('gui_images/temp.jpg')
      img_graph = StringProperty('gui_images/temp1.jpg')
      def __init__(self):
        super(LoginScreen, self).__init__()
        EventLoop.ensure_window()
        Window.clearcolor = (1, 1, 1, 1)
        Window.size = (1280,700)
        EventLoop.window.title = self.title = 'ball_detection'
        lbl1 = Label(text="Input Path : ",italic=True, bold=True, size_hint=(.08, .05) ,pos=(70,650), color = (0,0,0,0))
        lbl2 = Label(text="visualization :", bold=True, size_hint=(.08, .05) ,pos=(720,650), font_size='20sp', color = (0,0,0,0))
        txt1 = TextInput(multiline=False, font_size=20,size_hint=(.2, .05) ,pos=(180,650))
        ok = Button(text="OK",italic=True,size_hint=(.08, .05) ,pos=(450,650))
        stop = Button(text="Stop",italic=True,size_hint=(.08, .05) ,pos=(560,650))
        temp=cv2.imread('gui_images/img_original.jpg')
        temp=cv2.resize(temp, (450,450))
        cv2.imwrite('gui_images/temp.jpg',temp)
        img_graph = 'gui_images/temp.jpg'
        self.im_graph =  Image(source=img_graph, pos_hint={'center_x': 0.70, 'center_y': 0.58})
        in_px = Label(text="Results in CM/Pixels\n  Toss Height : \n  Toss Distance : \n  Toss Speed : \n  Ball Speed : ", bold=True,font_size=20, size_hint=(.2, .2) ,pos=(645,40), color =( 0,0,0,0))
        in_cm = Label(text="\n|\n|\n|\n|", bold=True,font_size=20, size_hint=(.2, .2) ,pos=(850,40), color =(0,0,0,0))
        str1 = str('\n')+str(0.0)+" px\n"+str(0.0)+" px\n"+str(0.0)+" px/sec\n"+str(0.0)+' px/sec'
        self.in_px_ans = Label(text=str1,italic=True, bold=True,font_size=20, size_hint=(.2, .2) ,pos=(780,40), color =(0,0,0,0))
        str2 = str('\n')+str(0.0)+" cm\n"+str(0.0)+" cm\n"+str(0.0)+" cm/sec\n"+str(0.0)+' cm/sec' 
        self.in_cm_ans = Label(text=str2,italic=True, bold=True,font_size=20, size_hint=(.2, .2) ,pos=(930,40), color =(0,0,0,0))
        img_src = 'gui_images/bg.png'
        self.im =  Image(source= img_src, pos_hint={'center_x': 0.28, 'center_y': 0.475})

        dropdown = DropDown()
        for index in range(1,15):
            btn = Button(text='Plot %d' % index, size_hint_y=None, height=42)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        mainbutton = Button(text='Show Plots' ,size_hint=(.1, .05) ,pos=(1140,605))
        mainbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: save_plot(x))

        self.add_widget(mainbutton)
        self.add_widget(self.im)
        self.add_widget(lbl1)
        self.add_widget(lbl2)
        self.add_widget(txt1)
        self.add_widget(ok)
        self.add_widget(stop)
        self.add_widget(in_px)
        self.add_widget(in_cm)
        self.add_widget(self.in_cm_ans)
        self.add_widget(self.in_px_ans)
        self.add_widget(self.im_graph)
        ok.bind(on_press=lambda *a:storage_for_file_name(txt1.text))
        stop.bind(on_press=lambda *a:storage_for_file_name('stop'))
        Clock.schedule_interval(self.update, 1.0 / 1000.0)
        

      def update(self,d):
        global cap,N,n,x,y,sc,height_px, width_px,top,safe_arr, toss_speed_px, flag_value, serve_speed_px, img, graph_update, plot_update, graph_count
        ret, image_np = cap.read()
        if ret: 
            image_np = cv2.resize(image_np,(600,600))
            image_safe = cv2.resize(image_np,(600,600))
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
            image_np, arr,score = visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores), {1: {'id': 1, 'name': u'ball'}})
            cv2.imwrite('gui_images/temp1.jpg',image_np)
            self.im.source = 'gui_images/temp1.jpg'
            self.im.reload()
            n+=1
            if arr != []:
              center_x = int((arr[2]+arr[3])/2)
              center_y = int((arr[0]+arr[1])/2)
              if center_x<350:
                x.append(center_x)
                y.append(center_y)
                N.append(n)
                sc.append(255-((int(score[6:-1])-50)*4))

              try:
                high_x = min(x)
                high_y = y[x.index(high_x)] 
                if high_x ==center_x:
                  top = image_safe[:]
                  safe_arr = arr
              except(ValueError):
                pass

            height_px, width_px ,toss_speed_px ,serve_speed_px, graph_update = start_recording()

            if graph_update:
              cm = 0.686
              graph_count+=1
              img = cv2.imread('gui_images/img.jpg')
              for i in range(len(x)):
                cv2.circle(img,(y[i], x[i]+50), 5, (0,sc[i], 255),-11)
              temp = cv2.resize(img, (450,450))
              cv2.imwrite('gui_images/img'+str(graph_count)+'.jpg',temp)
              self.im_graph.source = 'gui_images/img'+str(graph_count)+'.jpg'
              self.im_graph.reload()
              graph_update = 0
              x = []
              y = []
              sc = []
              N = []
              flag_value = 0
              # print round(height_px*cm,1), round(width_px*cm,1),round(toss_speed_px*cm,1),round(serve_speed_px*cm,1)
              height_cm, width_cm, toss_speed_cm, serve_speed_cm = round(height_px*cm,1), round(width_px*cm,1),round(toss_speed_px*cm,1),round(serve_speed_px*cm,1)
              self.in_cm_ans.text = str('\n')+str(height_px)+" px\n"+str(width_px)+" px\n"+str(toss_speed_px)+" px/sec\n"+str(serve_speed_px)+' px/sec'
              self.in_px_ans.text = str('\n')+str(height_cm)+" cm\n"+str(width_cm)+" cm\n"+str(toss_speed_cm)+" cm/sec\n"+str(serve_speed_cm)+' cm/sec'
              f = open('results.csv','a')
              li = str(graph_count)+','+str(height_px)+','+str(width_px)+','+ str(toss_speed_px)+','+str(serve_speed_px)+','+ str(height_cm)+','+ str(width_cm)+','+ str(toss_speed_cm)+','+ str(serve_speed_cm)
              f.write(li)
              f.write('\n')
              f.close


        if plot_update:
          img = cv2.imread('gui_images/img'+str(plot_update)+'.jpg')
          temp = cv2.resize(img, (450,450))
          cv2.imwrite('gui_images/temp.jpg',temp)
          self.im_graph.source = 'gui_images/temp.jpg'
          self.im_graph.reload()
          with open('results.csv', 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
          for i in lines:
            if i[0]==str(plot_update):
              self.in_cm_ans.text = str('\n')+str(i[1])+" px\n"+str(i[2])+" px\n"+str(i[3])+" px/sec\n"+str(i[4])+' px/sec'
              self.in_px_ans.text = str('\n')+str(i[5])+" cm\n"+str(i[6])+" cm\n"+str(i[7])+" cm/sec\n"+str(i[8])+' cm/sec'
          readFile.close()


    class ball_detection(App):
        def build(self):
          return LoginScreen()

    if __name__ == "__main__":
        ball_detection().run() 