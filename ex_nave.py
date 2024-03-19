import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math
from screeninfo import get_monitors
import pynput


def init_window(height, width):
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
    window = glfw.create_window(height, width, "Transformação Geométrica", None, None)
    glfw.make_context_current(window)

    return window

def init_shaders(vertex_code, fragment_code):
    # Request a program and shader slots from GPU
    program  = glCreateProgram()
    vertex   = glCreateShader(GL_VERTEX_SHADER)
    fragment = glCreateShader(GL_FRAGMENT_SHADER)

    # Set shaders source
    glShaderSource(vertex, vertex_code)
    glShaderSource(fragment, fragment_code)

    # Compile shaders
    glCompileShader(vertex)
    if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(vertex).decode()
        print(error)
        raise RuntimeError("Erro de compilacao do Vertex Shader")

    glCompileShader(fragment)
    if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(fragment).decode()
        print(error)
        raise RuntimeError("Erro de compilacao do Fragment Shader")

    # Attach shader objects to the program
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)

    # Build program
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        print(glGetProgramInfoLog(program))
        raise RuntimeError('Linking error')
        
    # Make program the default program
    glUseProgram(program)

    return program

def send_data_GPU(vertices, type_coord, program):

    # Request a buffer slot from GPU
    buffer = glGenBuffers(1)
    # Make this buffer the default one
    glBindBuffer(GL_ARRAY_BUFFER, buffer)

    # Upload data
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, buffer)

    # Bind the position attribute
    # --------------------------------------
    stride = vertices.strides[0]
    offset = ctypes.c_void_p(0)

    loc = glGetAttribLocation(program, "position")
    glEnableVertexAttribArray(loc)

    glVertexAttribPointer(loc, len(vertices[0][0]), type_coord, False, stride, offset)

def key_event(window,key,scancode,action,mods):
    global r_inc, s_inc, vel
    
    if key == 265: vel += 0.0001
    if key == 264: vel -= 0.0001
        
    if key == 65: r_inc += 0.1 # A
    if key == 83: r_inc -= 0.1 # D
        
    if key == 334: s_inc += 0.1 # Z
    if key == 333: s_inc -= 0.1 # X
        
    #print(key)

cursor_x = 0.0
cursor_y = 0.0

def poschange(x,y):
    global cursor_x, cursor_y
    w, h = get_screen_size()
    cursor_x = (x - w/2)/(w/2) #normalizando as coordenadas do cursor
    cursor_y = (h/2 - y)/(h/2)


def multiplica_matriz(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c

def get_screen_size():
    screen_width = 800
    screen_height = 800
    
    #Resolucao do monitor
    for m in get_monitors():
        screen_width = m.width
        screen_height = m.height
    
    return screen_width, screen_height

def calc_sin_cos(triang_x, triang_y):
    ca = cursor_x - triang_x
    co = cursor_y - triang_y
    hi = math.sqrt(ca**2 + co**2)

    if hi == 0:
        return 0, 0
    
    c = ca/hi
    s = co/hi

    return c, s

def calc_direcao_nave():
    global x_inc, y_inc

    if cursor_x > 0:
        x_inc = vel
    else:
        x_inc = vel*(-1)

    if cursor_y > 0:
        y_inc = vel
    else:
        y_inc = vel*(-1)
    
    return x_inc, y_inc


# translacao
x_inc = 0.0
y_inc = 0.0

# rotacao
r_inc = 0.0

# coeficiente de escala
s_inc = 1.0

#velocidade da nave
vel = 0.0

def main():

    screen_width, screen_height = get_screen_size()

    #screen_width = 800
    #screen_height = 800

    window = init_window(screen_width, screen_height)
    glfw.set_key_callback(window,key_event)

    vertex_code = """
            attribute vec2 position;
            uniform mat4 mat;
            void main(){
                gl_Position = mat * vec4(position,0.0,1.0);
            }
            """

    fragment_code = """
            void main(){
                gl_FragColor = vec4(1.0,0.0,0.0,1.0);
            }
            """

    program = init_shaders(vertex_code, fragment_code)

    # preparando espaço para 3 vértices usando 2 coordenadas (x,y)
    vertices = np.zeros(3, [("position", np.float32, 2)])

    # preenchendo as coordenadas de cada vértice
    vertices['position'] = [
                                ( 0.00, +0.05), 
                                (-0.05, -0.05), 
                                (+0.05, -0.05)
                            ]

    send_data_GPU(vertices, GL_FLOAT, program)

    glfw.show_window(window)

    t_x = 0.0
    t_y = 0.0
    angulo = 0.0
    s_x = 1.0
    s_y = 1.0

    triang_x = 0
    triang_y = 0

    #listener para captar movimentos do mouse
    mouli = pynput.mouse.Listener(on_move = poschange)
    mouli.start()

    while not glfw.window_should_close(window):

        t_x += x_inc
        t_y += y_inc
        angulo += r_inc
        
        s_x = s_inc
        s_y = s_inc

        calc_direcao_nave()

        #print(f"Posicao Triangulo:({triang_x},{triang_y})")
        
        c, s = calc_sin_cos(triang_x, triang_y)
        #c = math.cos( math.radians(angulo) ) #CA/HI
        #s = math.sin( math.radians(angulo) ) #CO/HI
        
        glfw.poll_events() 
        
        glClear(GL_COLOR_BUFFER_BIT) 
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        #Draw Triangle
        mat_rotation = np.array([  c  , -s , 0.0, 0.0, 
                                   s  ,  c , 0.0, 0.0, 
                                   0.0, 0.0, 1.0, 0.0, 
                                   0.0, 0.0, 0.0, 1.0], np.float32)
        
        mat_scale =    np.array([  s_x, 0.0, 0.0, 0.0, 
                                   0.0, s_y, 0.0, 0.0, 
                                   0.0, 0.0, 1.0, 0.0, 
                                   0.0, 0.0, 0.0, 1.0], np.float32)
        
        mat_translation = np.array([1.0, 0.0, 0.0, t_x, 
                                    0.0, 1.0, 0.0, t_y, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)
        
        
        mat_transform = multiplica_matriz(mat_translation,mat_rotation)
        mat_transform = multiplica_matriz(mat_transform,mat_scale)

        triang_x = mat_transform[0][3]
        triang_y = mat_transform[0][7]

        #Reposicionando a nave caso ela saia da telac
        if triang_x > 1:
            t_x = -1
        elif triang_x < -1:
            t_x = 1
        
        if triang_y > 1:
            t_y = -1
        elif triang_y < -1:
            t_y = 1

        #print(f"Posicao Cursor:({cursor_x},{cursor_y})")

  
        loc = glGetUniformLocation(program, "mat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)
        glDrawArrays(GL_TRIANGLES, 0, 3)


        glfw.swap_buffers(window)

    glfw.terminate()

main()