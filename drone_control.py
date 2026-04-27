#!/usr/bin/env python3
import rospy
import tf
import math as m
import numpy as np
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import TwistStamped
import sympy as sp
from scipy.linalg import pinv
import matplotlib.pyplot as plt
from collections import deque
from mavros_msgs.msg import AttitudeTarget
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as Rdos
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

#CONTROLLER GAINS
rho1 = 8


k1_x = 1 
k1_y = 1 
k1_z = 8

k2_x = 2 
k2_y = 2 
k2_z = 1

k1_diag = np.diag([k1_x, k1_y, k1_z])
k2_diag = np.diag([k2_x, k2_y, k2_z])


rho2 = 12 #2
xi_subkappa = np.zeros((3,1))
xi_subkappa_dot= np.zeros((3,1))



#OBSTACLE GAIN 
gamma = 2

#LOW PASS FILTER CONSTANTS 
miu1= 0.6
miu2 = 0.6
Q_star = np.zeros((6,1))
prev_vec = np.array([1.0, 0.0])
psi_accum = 0


#CONTROLLER MATRICES 
D_inv = np.zeros((6, 6))
D = np.zeros((6, 6))
drag_matrix = np.zeros((6, 6))
g_q = np.zeros((6, 1))
I_zero = np.zeros((3, 6))
zero_I = np.zeros((3, 6))
l_arm = np.zeros((1, 1))


# Constantes THETA 1 Y 2 
theta1 =  3.0 #0.1 
theta2 =  5.0 # 0.5
eta1= np.zeros((3,1))
eta2 =np.zeros((3,1))
epsilon = pow(1,-6)
# contantes de ETA1 

init = False 
kappa_0 = np.zeros((3,1))
v_0 = np.zeros((3,1))
kappa_desired_0 = np.zeros((3,1))
v_deseada_0 = np.zeros((3,1))
omega = np.zeros((3,1))  # inicial
phi_d = 0
theta_d = 0

#CONSTANTES ETA 2 

w = 0.07   # CONSTANTES TRAJECTORIA DESEADA ELIPSE 
a = 2
b = 2
x0 = 0
y0 = 0
z0 = 0
theta_x = 0   # Angulo de inclinacion en x en la trayectoria 
theta_y = 0 #m.pi/6 #angulo de inclinacion en y 

vx = 0    # VELOCIDADES LINEALES 
vy = 0
vz = 0
wx = 0    # VELOCIDADES ANGULARES
wy = 0
wz = 0
v_actual = np.zeros((3,1))

kappa_x = 0
kappa_y = 0
kappa_z = 0
kappa = np.zeros((3,1))
q_x=0
q_y=0
q_z=0
q_w=0
roll = 0
pitch = 0
yaw = 0

theta = np.zeros((3,1))
theta_dot  = np.zeros((3,1))

yaw_d= 0
psi_dot_d = 0
# INTEGRAL A1

int_a1x =0 
a1_averagedx = 0

int_a1y =0 
a1_averagedy = 0

int_a1z =0 
a1_averagedz = 0

a1_averaged = np.zeros((3,1))
int_a1_averaged = 0

# INTEGRAL A2
int_a2x =0 
a2_averagedx = 0

int_a2y =0 
a2_averagedy = 0

int_a2z =0 
a2_averagedz = 0

a2_averaged = np.zeros((3,1))
t_prev = None

value_s1 = np.zeros(3)
value_s2 = np.zeros(3)
R = np.zeros((3,3))

# YAW
yaw_d_prev = 0.0  # yaw deseado anterior
max_yaw_rate = np.radians(20)  # 15 límite similar a PX4

collision = 0


# REAL position RVIZ 
current_pose = None
new_odom = False


# ANTI WINDUP
# Límites para integrales de posición (a1)
LIMIT_INT_A1X = 20.0
LIMIT_INT_A1Y = 5.0
LIMIT_INT_A1Z = 8.0
LIMIT_INT_A  = 8.0 #intregral de a1_averaged

# Límites para integrales de orientación (a2)
LIMIT_INT_A2X = 1.5   # Roll
LIMIT_INT_A2Y = 1.5   # Pitch  
LIMIT_INT_A2Z = 0.8   # Yaw

# También añade límites adaptativos opcionales
ENABLE_ADAPTIVE_LIMITS = True

def yaw_generator(vx, vy):
    global prev_vec, psi_accum

    curr_vec = np.array([vx, vy])
    norm = np.linalg.norm(curr_vec)

    if norm < 1e-6:
        curr_vec = prev_vec
    else:
        curr_vec = curr_vec / norm

    cross_z = prev_vec[0]*curr_vec[1] - prev_vec[1]*curr_vec[0]
    dot_val = prev_vec[0]*curr_vec[0] + prev_vec[1]*curr_vec[1]

    delta_psi = np.arctan2(cross_z, dot_val)

    psi_accum += delta_psi
    prev_vec = curr_vec
    return psi_accum.item()


def ang_error(a, b):
    e = a - b
    return (e + np.pi) % (2*np.pi) - np.pi

def fcn(Ka):
    global D_inv, D, drag_matrix, g_q, l_arm, I_zero, zero_I , hover , Ixx, Iyy, Izz
    M = 1.5      # drone mass
    hover = 0.7
    g = 9.81       #  [m/s²]
    Ixx, Iyy, Izz = 0.029125, 0.029125, 0.055225  # Inertia [kg*m²]

    D = np.diag([M, M, M, Ixx, Iyy, Izz])
    D_inv = pinv(D)
    C = np.zeros((6,6))
    g_q = np.array([[0],[0],[M*g],[0],[0],[0]])
    
    print("----------Gq shape-------", g_q.shape)
    
    # Matrices de selección
    I_zero = np.hstack([np.eye(3), np.zeros((3, 3))])
    zero_I = np.hstack([np.zeros((3, 3)), np.eye(3)])
    
    # Matriz de arrastre
    drag_matrix = np.vstack([
        np.hstack([np.diag([Ka, Ka, Ka]), np.zeros((3, 3))]),
        np.zeros((3, 6))
    ])
    
    return D, D_inv, C, g_q, I_zero, zero_I, drag_matrix, l_arm, M, g

def odom_cb(msg):
    global kappa, v_actual, theta, theta_dot, new_odom
    new_odom = True
    # posición
    kappa = np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ]).reshape(3,1)

    # velocidad WORLD
    v_actual = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z
    ]).reshape(3,1)

    q = msg.pose.pose.orientation
    quat = [q.x, q.y, q.z, q.w]

    R_bw = tf.transformations.quaternion_matrix(quat)[0:3,0:3]

    v_msg = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z
    ]).reshape(3,1)

    v_actual = v_msg

    # orientación
    q = msg.pose.pose.orientation
    roll, pitch, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

    theta = np.array([roll, pitch, yaw]).reshape(3,1)

    # velocidad angular BODY
    theta_dot = np.array([
        msg.twist.twist.angular.x,
        msg.twist.twist.angular.y,
        msg.twist.twist.angular.z
    ]).reshape(3,1)

def laser_cb(msg):
    global distance_min, collision, x_obs, y_obs, angle_obs, t
    laser_time = rospy.Time.now().to_sec()
    ranges = np.array(msg.ranges) # Lista de ros a numpy array 
    ranges = np.where(np.isfinite(ranges), ranges, np.inf) 
    distance_verif= np.min(ranges)
    if np.isinf(distance_verif):
        #print('--------No obstacle detected---------')
        x_obs = 100
        y_obs = 10
        distance_min = 100
        collision = 0
    elif distance_verif < 1.5:
        collision = 1
        distance_min = np.min(ranges)
        #print('--------Possible Collision detected ---------', laser_time)
        idx = np.argmin(ranges)
        angle_obs = msg.angle_min + idx * msg.angle_increment
        #print("------ANGLE OBSTACLE -----", angle_obs)
        x_obs = distance_min * np.cos(angle_obs) #coordenadas del obstaculo x
        y_obs = distance_min * np.sin(angle_obs) #coordenadas del obstaculo y


def main():
    global kappa, theta, v_actual, theta_dot, kappa_0, theta_0, v_0, theta_dot_0, kappa_desired_0, v_deseada_0, init, eta1, eta2, wx, wy, wz, a1_averaged, t_prev
    global int_a1x, int_a1y, int_a1z, int_a2x, int_a2y, int_a2z, value_s1, value_s2, Q_star, yaw_d_prev, yaw_d, psi_dot_d, int_a1_averaged, omega
    global distance_min, collision, x_obs, y_obs, angle_obs, new_odom, phi_d, theta_d
    Ka = 0.1  # coeficiente de arrastre DRAG 
    D, D_inv, C, g_q, I_zero, zero_I, drag_matrix, l_arm, M, g = fcn(Ka)

    rospy.init_node("trajectory_pub")
    pub_att = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
    pub_path = rospy.Publisher("/desired_path", Path, queue_size=10)
    pub_real_path = rospy.Publisher("/real_path", Path, queue_size=10)
    rospy.Subscriber("/iris/scan", LaserScan, laser_cb)
    rospy.Subscriber("/mavros/local_position/odom", Odometry, odom_cb)
    # DESIRED PATH RVIZ
    path_msg = Path()
    path_msg.header.frame_id = "odom"
    # REAL PATH RVIZ 
    real_path_msg= Path()
    real_path_msg.header.frame_id = "odom"
    
    rate = rospy.Rate(150)
    rospy.sleep(1)
    

    #ATTITUDE MESSAGE
    att_msg = AttitudeTarget()
    att_msg.type_mask = 7   # ignorar body rates
    
    print("------------publishing set points-------------")
    for _ in range(40):
        att_msg.orientation.w = 1.0
        att_msg.thrust = 0.5
        pub_att.publish(att_msg)
        rate.sleep()

    rospy.wait_for_service("/mavros/set_mode")
    rospy.wait_for_service("/mavros/cmd/arming")
    try:
        set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        set_mode(base_mode=0, custom_mode="OFFBOARD")
        print("------------OFFBOARD--------------")

        arm(True)
        print("------------ARMING--------------")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        print("------------SERVICE FAILED--------------")

    rospy.wait_for_message("/mavros/local_position/velocity_local", TwistStamped)
    rospy.wait_for_message("/mavros/local_position/pose", PoseStamped)
    t_control_inicio = False 

    while not rospy.is_shutdown():

        if not new_odom: #si no hay medicion de odometria, no calculas control 
            rate.sleep()
            continue #Saltar todo lo que queda del loop y volver al inicio
        new_odom = False

        if not t_control_inicio:
            t_inicio = rospy.Time.now()
            t_control_inicio = True

        t = (rospy.Time.now() - t_inicio).to_sec()


        # POSICION DESEADA 
        x_t = a*m.cos(w*t)
        y_t = b*m.sin(w*t)
        z_t = 1.5

        # VELOCIDAD DESEADA 
        dx = -a*w*m.sin(w*t)
        dy = b*w*m.cos(w*t)
        dz = 0
        v_desired = np.array([dx, dy,dz]).reshape(3,1)

        # ACELERACIONES 
        ddx = -a*w*w*np.cos(w*t)
        ddy = -b*w*w*np.sin(w*t)
        ddz = 0
        v_desired_dot = np.array([ddx, ddy,ddz]).reshape(3,1)

        # VELOCIDAD ANGULAR DESEADA EN YAW
        psi_dot_d = (dx*ddy - dy*ddx)/((dx*dx + dy*dy)+epsilon)

        #ATTITUDE MESSAGE
        att_msg = AttitudeTarget()
        att_msg.type_mask = 7 

        # matrices de rotacion
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        Rtraj = Ry @ Rx
        rotated_point = Rtraj @ np.array([x_t,y_t,z_t]).reshape(3,1)
        rotated_vel   = Rtraj @ np.array([dx,dy,dz]).reshape(3,1)

        if not init:
            kappa_0 = kappa   
            v_0 = v_actual    
            kappa_desired_0 = (rotated_point + kappa_0).reshape(3,1)
            v_deseada_0 = rotated_vel.reshape(3,1)
            theta_0 = theta
            theta_dot_0 = theta_dot
            theta_desired_0 = np.array([0, 0, yaw_d]).reshape(3,1)
            theta_dot_desired_0 = np.array([0, 0, psi_dot_d]).reshape(3,1)
            eta1 = (-theta1*(v_0 - v_deseada_0) + kappa_desired_0 - kappa_0).reshape(3,1)
            sum_eta1 = eta1 + theta1*(v_0 - v_deseada_0)-kappa_desired_0 + kappa_0

            eta2 = (-theta2*(theta_dot_0 - theta_dot_desired_0) + theta_desired_0-theta_0).reshape(3,1)
            sum_eta2 = eta2 + theta2*(theta_dot_0 - theta_dot_desired_0)-theta_desired_0 + theta_0
            print('-----SUMA ETA 1 ----', sum_eta1)
            print('-----SUMA ETA 2 ----', sum_eta2)
            init = True

        kappa_desired = (rotated_point+ kappa_0 ).reshape(3,1) #+ kappa_0
        # RVIZ  msg reference
        posed = PoseStamped()
        posed.header.frame_id = "odom"
        posed.header.stamp = rospy.Time.now()

        posed.pose.position.x = kappa_desired[0,0]
        posed.pose.position.y = kappa_desired[1,0]
        posed.pose.position.z = kappa_desired[2,0]

        posed.pose.orientation.w = 1.0

        # ===== TRAYECTORIA REAL ===== RVIZ
        real_pose = PoseStamped()
        real_pose.header.frame_id = "odom"
        real_pose.header.stamp = rospy.Time.now()

        real_pose.pose.position.x = kappa[0,0]
        real_pose.pose.position.y = kappa[1,0]
        real_pose.pose.position.z = kappa[2,0]

        real_pose.pose.orientation.w = 1.0


        #---------------------------- PUBLICAMOS 
        path_msg.header.stamp = rospy.Time.now()
        real_path_msg.header.stamp = rospy.Time.now()


        max_len = 1100
        path_msg.poses.append(posed)
        if len(path_msg.poses) > max_len:
            path_msg.poses.pop(0)

        real_path_msg.poses.append(real_pose)
        if len(real_path_msg.poses) > max_len:
            real_path_msg.poses.pop(0)

        pub_path.publish(path_msg)
        pub_real_path.publish(real_path_msg)


        #------------DELTA 1, DELTA1_DOT 
        delta1 = (kappa - kappa_desired).reshape(3,1) 
        #delta1[1]*=2
        #delta1[2]*=2
        delta1_dot = (v_actual - v_desired).reshape(3,1)
        #delta1_dot[1] *= 4.0   # increases damping over Z
        #delta1_dot[2] *= 4.0 

        F_d1 = np.linalg.norm(delta1)
        a1 = -delta1.reshape((3,1))  if F_d1 >= epsilon else np.zeros((3,1))
        #a1[2] = -1.5*delta1[2]   # más fuerte en Z

        #------------     DELTA 2---------------------------------- 
        theta_desired = np.array([phi_d, theta_d, yaw_d]).reshape(3,1)
        theta_dot_desired = np.array([0,0,psi_dot_d]).reshape(3,1)

        delta2 = theta - theta_desired
        delta2_dot = theta_dot - theta_dot_desired

        F_d2 = np.linalg.norm(delta2)
        a2 = -delta2.reshape((3,1)) if F_d2 >= epsilon else np.zeros((3,1)) 

        if t_prev is None:
            t_prev = t

        dt = t - t_prev
        #dt = 1.0 / 150.0   # si odometría es ~30 Hz   se revisa con  rostopic hz /mavros/local_position/odom

        t_prev = t
        #print("dt control:", dt)

        int_a1x += a1[0].item()*dt # int_a1x += a1[0].item()*dt
        int_a1y += a1[1].item() * dt
        int_a1z += a1[2].item() * dt

        int_a1x = np.clip(int_a1x, -LIMIT_INT_A1X, LIMIT_INT_A1X)
        int_a1y = np.clip(int_a1y, -LIMIT_INT_A1Y, LIMIT_INT_A1Y)
        int_a1z = np.clip(int_a1z, -LIMIT_INT_A1Z, LIMIT_INT_A1Z)

        a1_averagedx = int_a1x/(t + theta1)
        a1_averagedy = int_a1y/(t + theta1)
        a1_averagedz = int_a1z/(t + theta1)

        a1_averaged = np.vstack([a1_averagedx, a1_averagedy, a1_averagedz]).reshape(3,1)
        
        int_a1_averaged += a1_averaged*dt
        #int_a1x = np.clip(int_a1x, -LIMIT_INT_A1X, LIMIT_INT_A1X)
        #A2 Averaged 
        int_a2x += a2[0].item() * dt
        int_a2y += a2[1].item() * dt
        int_a2z += a2[2].item() * dt

        int_a2x = np.clip(int_a2x, -LIMIT_INT_A2X, LIMIT_INT_A2X)
        int_a2y = np.clip(int_a2y, -LIMIT_INT_A2Y, LIMIT_INT_A2Y)
        int_a2z = np.clip(int_a2z, -LIMIT_INT_A2Z, LIMIT_INT_A2Z)        

        a2_averagedx = int_a2x / (t + theta2)
        a2_averagedy = int_a2y / (t + theta2)
        a2_averagedz = int_a2z / (t + theta2)
        a2_averaged = np.vstack([a2_averagedx, a2_averagedy, a2_averagedz]).reshape(3,1)
        
        # Sliding surfaces 
        tau1 = theta1 #max(t+theta1, theta1)  # mínimo 0.5s   tenia 5.0 25/03 9.43pm   0.5 26/03 9.49pm
        tau2 = theta2 #max(t+theta2, theta2)
        s1 = delta1_dot + (delta1 + eta1)/tau1 - a1_averaged
        s2 = delta2_dot + (delta2 + eta2)/tau2 - a2_averaged

        fv = -I_zero@D_inv@(C+drag_matrix)@np.vstack([v_actual, theta_dot]).reshape(6,1)-I_zero@ D_inv@g_q
        fw= -zero_I@D_inv@(C+drag_matrix)@np.vstack([v_actual, theta_dot]).reshape(6,1)-zero_I@D_inv@g_q
        hv=fv-v_desired_dot+(v_actual-v_desired)/(t+theta1) - a1/(t+theta1) - (delta1+eta1)/(t+theta1)**2 + int_a1_averaged/(t+theta1)**2 #a1_averaged
        hw= fw+ theta_dot/(t + theta2) - a2/(t+theta2) - (delta2+eta2)/(t+theta2)**2 + a2_averaged /(t+theta2)**2
        #------------- U CONTROLLER -------------------------------------

        phi1 = 0.4 # 0.4, 0.3
        phi1_x = 0.2
        phi2 = 0.4

        #value_s1 = np.array([s1[i,0]/(abs(s1[i,0])+phi1) for i in range(3)]).reshape(3,1)
        value_s1 = np.array([
            s1[0,0] / (abs(s1[0,0]) + phi1_x),
            s1[1,0] / (abs(s1[1,0]) + phi1),
            s1[2,0] / (abs(s1[2,0]) + phi1)
        ]).reshape(3,1)

        value_s2 = np.array([s2[i,0]/(abs(s2[i,0])+phi2) for i in range(3)]).reshape(3,1)
        #k1 =  d+ ∥h∥ + M g + c+ ∥q∥ + ξ+ + ρ,
        #k1 = np.linalg.norm(xi_subkappa)/(t+theta1) + np.linalg.norm(xi_subkappa_dot)+ rho1
        h1 = -kappa_desired + (delta1)/(t+theta1) - (delta1 + eta1)/(t+theta1)**2  - (1/(t+theta1))*a1_averaged + a1_averaged 
        h1_norm = np.linalg.norm(h1)
        h2 = -theta_desired + (delta2)/(t+theta2) - (delta2+ eta2)/(t+theta2)**2  - (1/(t+theta2))*a2_averaged + a2_averaged 
        h2_norm = np.linalg.norm(h2)

        k1 = 1*(M+ h1_norm +M*9.81+ np.linalg.norm(kappa) + rho1)
        k2 = 1*(M + h2_norm +np.linalg.norm(theta)+ rho2)


        u1 = D@(-np.vstack([hv, hw]) -np.vstack([k1*value_s1,k2*value_s2]))
        #u1 = D@(-np.vstack([hv, hw]) -np.vstack([k1_diag@value_s1,k2_diag@value_s2]))

        # low pass filter 
        Q_star[0:3] = (1 - miu1)*Q_star[0:3] + miu1*u1[0:3]
        Q_star[3:6] = (1 - miu2)*Q_star[3:6] + miu2*u1[3:6]

#-----------------------------------------------------------------------------
        Fd = Q_star[0:3].reshape(3,1)
        tau = Q_star[3:6].reshape(3,1)
        print('TAU ',tau)
        tau = np.clip(tau, -0.1, 0.1)
        print('TAU SAT ',tau)

        phi_d   = (1/g) * (Fd[0].item()*np.sin(yaw_d) - Fd[1].item()*np.cos(yaw_d))
        theta_d = (1/g) * (Fd[0].item()*np.cos(yaw_d) + Fd[1].item()*np.sin(yaw_d))
        phi_d = np.clip(phi_d, -0.3, 0.3)
        theta_d = np.clip(theta_d, -0.3, 0.3)


        I = np.diag([Ixx, Iyy, Izz])
        I_inv = np.linalg.inv(I)
        omega_cross = np.cross(omega.flatten(), (I @ omega).flatten()).reshape(3,1)
        omega_dot = I_inv @ (tau - omega_cross - 0.1 * omega)
        
        omega = omega + omega_dot * dt
        omega = np.clip(omega, -1.0, 1.0)  # rad/s (ajusta)
        

        thrust = Fd[2].item() / (M * g)
        thrust = np.clip(thrust, 0.0, 1.0)

        print('Fd', Fd)
        print('Kappa', kappa)
        print('Kappa deseada', kappa_desired)
        print('Delta 1', delta1)
        print('Delta 2', delta2)
        print('s1', s1)
        print('s2', s2)
        print('\n')
        # --- msg AttitudeTarget ---
        att_msg = AttitudeTarget()
        att_msg.type_mask =128 #(AttitudeTarget.IGNORE_ATTITUDE)
        
        att_msg.body_rate.x = omega[0].item()
        att_msg.body_rate.y = omega[1].item()
        att_msg.body_rate.z = omega[2].item()
        att_msg.thrust = thrust

        pub_att.publish(att_msg)
        rate.sleep()
#-----------------------------------------------------------------------------
        
if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass
