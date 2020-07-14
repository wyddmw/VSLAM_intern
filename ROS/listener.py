#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import String
from turtlesim.msg._Pose import Pose                # 在这个地方需要引入消息的数据类型
# from geometry_msgs.msg import Vector3

def callback(data):
    # 回调函数，如果订阅的topic有符合的数据类型发出，就会执行这个函数，具体要使用data哪个数据要查看该数据类型都包含哪些数据成员
    # 如果是Pose这个数据类型的data，包含的数据有data.x data.y等，可以通过查看data这个类进行了解
    rospy.loginfo(rospy.get_caller_id() + "I heard %f", data.x)
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    # 初始化一个节点，命名为listener
    rospy.init_node('listener', anonymous=True)
    # 订阅topic turtle1/pose
    rospy.Subscriber("/turtle1/pose", Pose, callback)

    # 进入自循环
    rospy.spin()

if __name__ == '__main__':
    listener()