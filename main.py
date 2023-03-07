import os
import cv2
import pcl
import math
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks

import numpy.polynomial.polynomial as poly


show = True
plotcount = 0

class Lane:
    '''  
    Detect lane points by using a sliding window search where the initial estimates for the sliding windows are made using an intensity-based histogram
    '''

    def peak_intensity_ratio(self,ptCloud,bin_size):  
        '''
        creates a histogram of intensity points. control the number of bins for the histogram by specifying the bin_size
        '''
        y=ptCloud[:,1]
        min_y=math.ceil(y.min())
        max_y=math.ceil(y.max())
        y_val=np.linspace(min_y,max_y,bin_size)
        avg_intensity=[]
        ymean=[]
        for i in range(len(y_val)-1):
            index=self.get_index_inrange(y,y_val[i],y_val[i+1])
            intensity_sum=0
            for j in index:
                intensity_sum+=ptCloud[j,3]

            avg_intensity.append(intensity_sum)
            ymean.append((y_val[i]+y_val[i+1])/2) # Detect lane points by using a sliding window search, where the initial estimates for the sliding windows are made using an intensity-based histogram.
        
        plt.plot(ymean,avg_intensity,'--k')
        return ymean,avg_intensity

    def get_index_inrange(self,arr,start,end):
        ''' gets data within start :to ed range '''
        ind=[i for i in range(len(arr)) if arr[i]>=start and arr[i]<end]
        return ind

    def lane_find_peaks(self,a):
        '''Obtain peaks in the histogram'''
        x = np.array(a)
        ret = []
        peaks, _ = find_peaks(x)
        print("Peaklen="+str(len(peaks)))
        print(peaks)
        lis = (sorted(range(len(peaks)), key=lambda i: x[peaks[i]])[-3:])
        ret = [peaks[i] for i in lis]
        # print([peaks[i] for i in lis])
        # for i in range(len(peaks)):
        #     print(x[peaks[i]])
        #     ret.append(peaks[i])
        return np.array(ret)

    def lane_find_peaks0(self,a):
        '''Obtain peaks in the histogram'''
        x = np.array(a)
        max = np.max(x)
        length = len(a)
        ret = []
        for i in range(length):
            ispeak = True
            if i-1 > 0: # Detect lane points Contains Helper class forby using a sliding window search, where the initial estimates for the sliding windows are made using an intensity-based histogram.
                if ispeak:
                    ret.append(i)
        return np.array(ret)

    def ransac_polyfit(self,x, y, order=2, n=20, k=100, t=0.1, d=100, f=0.8):
        '''
            polynomial fitting 
            # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
            
            # n – minimupm number of data points required to fit the model
            # k – maximum number of iterations allowed in the algorithm
            # t – threshold value to determine when a data point fits a model
            # d – number of close data points required to assert that a model fits well to data
            # f – fraction of close data points required
        '''
        besterr = np.inf
        bestfit = None
        for kk in range(len(x)):
            maybeinliers = np.random.randint(len(x), size=n)
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
            alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
            if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
                bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
                thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
        return bestfit

    # def DisplayBins(self,x_val,y,color):

    #     y_val=[y]*len(x_val)
    #     plt.plot(x_val,y_val,c=color)

    

    def DetectLanes(self,data,hbin,vbin, start,min_x,max_x,num_lanes):
        '''
            sliding window approach to detects lane points
        '''
        verticalBins = np.zeros((vbin, 4, num_lanes))
        lanes = np.zeros((vbin, 4, num_lanes))
        # verticalBins=[]
        # lanes=[]
        laneStartX = np.linspace(min_x,max_x, num=vbin)
        
        startLanePoints=start
        
            # print('for index i',i)
            # print('after each updation',startLanePoints)
        for i in range(vbin - 1):
            for j in range(num_lanes):

                laneStartY = startLanePoints[j]
                lowerbound = laneStartY - hbin # Added
                upperbound = laneStartY + hbin # Added
                # print('starting x',laneStartX [i]Detect lane points by using a sliding window search, where the initial estimates for the sliding windows are made using an intensity-based histogram.

                inds = np.where((data[:,0] < laneStartX[i+1] )& (data[:,0] >= laneStartX[i]) & (data[:,1] < upperbound) & (data[:,1]>=lowerbound))[0]
                
                # print(len(inds))
                # plt.scatter(data[inds,0],data[inds,1],c='yellow')

                if len(inds)!=0:
                    # plt.vlines(laneStartX[i],-15,15)
                    roi_data=data[inds,:]
                    max_intensity=np.argmax(roi_data[:,3])
                    val=roi_data[max_intensity,:]
                    verticalBins[i,:,j]=val
                    lanes[i,:,j]=val
                    startLanePoints[j]=roi_data[max_intensity,1]
                    plt.scatter(roi_data[max_intensity,0],roi_data[max_intensity,1],s=10,c='yellow')
                    # plt.plot(roi_data[max_intensity,0],roi_data[max_intensity,1],color='green')
                 
        return lanes

def render_lanes_on_image(data,img, calib, img_width, img_height,figg):
    """
    Overlay lane lines on the original frame
    """

    print('data in lane_image function',len(data))
    proj_velo2cam2 = project_velo_to_cam2(calib)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    
    
    # for i in range(data.shape[2]):
    #     d=data[:,:,i]
    for d in data:
        pts_2d = project_to_image(d.transpose(), proj_velo2cam2)
        inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] > 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1,:]>0)  )[0]

        # print(inds)

        # Filter out pixels points
        imgfov_pc_pixel = pts_2d[:, inds]

        # Retrieve depth from lidar
        imgfov_pc_velo = d[inds, :]
        # imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
        imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
        # Create a figure. Equal aspect so circles look circular  
        # Show the image
        ax.imshow(img)
        ax.plot(imgfov_pc_pixel[0],imgfov_pc_pixel[1],color='red',linewidth=2)
    
    plt.savefig('output/'+figg+'.png')
    plt.show()
    # show(img)
    
    # return imgfov_pc_pixel[0], imgfov_pc_pixel[1]


def fit_polynomial(lanes,peaks):
    ''' 
        estimates a polynomial curve on the detected lane points 
    '''
    polynomial_features = PolynomialFeatures(degree = 2)
    l=[]
    
    for i in range(len(peaks)):
        # print(lanes[:,:,i].shape)
        lane=lanes[:,:,i]
        # lane=[lane[i,:] for i in range(len(lane)) if len(lane[i])>0]
        
        # X_TRANSF=np.reshape(lane[:,0],(-1,1))
        # y=np.reshape(lane[:,1],(-1,1))
        lane=[lane[i,:] for i in range(len(lane)) if lane[i,0]!=0 and lane[i,1]!=0]
        lane=np.array(lane).reshape(-1,4)
        x=lane[:,0]
        y=lane[:,1]
       
        if len(x)>0 and len(y)>0:
            coefs =  poly.polyfit(x,y,2)
            # X_NEW=np.linspace(x.min()+5,x.max(),70)
            X_NEW=x
            ffit = poly.Polynomial(coefs)
            Y_NEW=ffit(X_NEW)
            # print('for each lane min depth',lane[:,2].min())
            z=np.ones(len(X_NEW))*(lane[:,2].mean())
            # print('lane[:,2].min()',lane[:,2].min())
            # for i in range(len(X_NEW)):
            #     new=(lane[:,2].min())/(len(X_NEW)-i+1)
            #     z[i]=new 

            # # print('mean',lane[:,2].mean())
            intensity=np.ones(len(X_NEW))*lane[:,3].mean()

            # lane[:,2]=lane[:,2]-3

            point1=np.concatenate((X_NEW.reshape(-1,1),Y_NEW.reshape(-1,1)),axis=1)
            point2=np.concatenate((lane[:,2].reshape(-1,1),intensity.reshape(-1,1)),axis=1)

            newpoints=np.concatenate((point1,point2),axis=1)
            l.append(newpoints)

            # print(abs(Y_NEW-y))
        
        # break
    
    return l
    # return np.array(l).reshape(-1,4,len(peaks))


def remove_noise(data):
    '''
    Fit the Data and filter out noise 
    '''
    # X = [i for i in zip(['x'],p['y'])]
    X = StandardScaler().fit_transform(data[:,0:3])
    db = DBSCAN(eps=0.05, min_samples=10).fit(X)
    # # print(type(db))
    db_labels = db.labels_
    pc=pd.DataFrame()
    pc['x']=data[:,0]-5
    pc['y']=data[:,1]
    pc['z']=data[:,2]
    pc['Intensity']=data[:,3]
    pc['labels']=db_labels
    pc['r'] = np.sqrt(pc['x'] ** 2 + pc['y'] ** 2)

    #   remove noisy point clouds data
    labels, cluster_size = np.unique(pc['labels'], return_counts=True)
    # pc = pc[pc['labels']>=0] 

    meanz=pc["z"].mean()
    stdz = pc["z"].std()
    minz=pc["z"].min()

    pc=pc[pc["z"] < meanz + 4*stdz ]

    print(meanz,minz)


    # shift plane to ground 
    pc['z']=pc['z']-4

    return pc

def read_lidar_data(data_path):
    ''' reads point cloud data '''
    pointcloud = np.fromfile(str(data_path), dtype=np.float32, count=-1).reshape([-1,4])
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    I = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    data=pd.DataFrame()
    data['x']=x
    data['y']=y
    data['z']=z
    data['Intensity']=I
    return data,pointcloud

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary'''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def project_velo_to_cam2(calib):
    ''' Create a projection matrix 
    Args: 
        calib: calibration data 
    '''
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_to_image(points, proj_mat):
    '''
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    '''
    num_pts = points.shape[1]
    points = proj_mat @ points
    
    points[:2, :] /= points[2, :]
    # points[:2, :]=np.nan_to_num(points[:2, :]) 
    return points[:2, :]

def render_lidar_on_image(pts_velo, img, calib, img_width, img_height,label):
    '''
        Overlay pointcloud data on the original rgb frame
    '''
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) & (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) & (pts_velo[:, 0] > 0))[0]
    print("inds:", inds.shape)
    print("pts_2d:", pts_2d.shape)
    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    # imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    # Create a figure. Equal aspect so circles look circular
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(img, extent=[0, img_width, img_height, 0])
    ax.scatter(imgfov_pc_pixel[0], imgfov_pc_pixel[1],s=3,c=label[inds])
    # ax.label()
    # print(len(label[inds]))
    plt.yticks([])
    plt.xticks([])
    plt.show()
    # return 0,0,0,0
    return imgfov_pc_pixel[0], imgfov_pc_pixel[1],pts_velo[inds,2],inds

def find_road_plane(points):
    ''' RANSAC Algorithm to segment ground plane '''
    cloud = pcl.PointCloud_PointXYZI()
    cloud.from_array(points.astype('float32'))
    fil = cloud.make_passthrough_filter()
    fil.set_filter_field_name("z")
    # print(points[:,2].min(),points[:,2].mean(),points[:,2].max())
    fil.set_filter_limits(points[:,2].min(),0)
    cloud_filtered = fil.filter()
    #  create a pcl object 
    seg =  cloud_filtered.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.8)
    indices, model = seg.segment()
    cloud_plane = cloud.extract(indices, negative=False)
    return cloud_plane.to_array(), np.array(indices)

def running_mean(x, N):
    """ x == an array of data. N == number of samples per average """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    


def show(filename):
    # Visualize
    # filename = cv2.imread('dataset/data_road/training/image_2/um_000000.png')
    cv2.imshow("Sample",filename)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def remove_shortfits(arr_list):
    new_arr = []
    for arr in arr_list:
        if(len(arr)>=5):
            new_arr.append(arr)
    return new_arr

def main():
    filenames=['um_000069.png']
    # filenames = os.listdir('dataset/data_road/training/image_2')
    # print(len(filenames))
    for filename in filenames:
        filename=filename.replace('.png',"")
        rgb = cv2.imread('dataset/data_road/training/image_2/'+ filename+'.png')
        h, w, c = rgb.shape
        data, lidar = read_lidar_data("dataset/data_road_velodyne/training/velodyne/"+filename+".bin")
        calib = read_calib_file('dataset/data_road/training/calib/'+filename+'.txt')
        render_lidar_on_image(lidar,rgb, calib,w,h,lidar[:,2])
        data=data.to_numpy()
        cloud,ind=find_road_plane(data)
        data=remove_noise(cloud)
        data=data.to_numpy()
        print('after road plane and noise removal',len(data))
        # print(data)
        # return 1
        print("--------------")
        # print(lidar)
        # render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,2])
        # show(rgb)
        # ###############  finding the lanes ############
        plt.figure()
        lane=Lane()
        yval,histVal=lane.peak_intensity_ratio(data,50)
        # peaks= find_peaks(histVal)[0]
        peaks=lane.lane_find_peaks(histVal)
        # print("histVal: "+ ",".join(str(histVal)))
        print("peaks: "+ np.array2string(peaks, separator=', '))
        # print("peaks: "+ ",".join([str(yval[x]) for x in peaks]))
        print(len(peaks))
        mid=int(len(peaks)/2)
        print('starting lane points',mid)
        ############### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # peaks=peaks[2:7]
        # peaks=peaks[mid-1:mid+1]
        # print("peaks: "+ np.array2string(peaks, separator=', '))
        # try:
        #     peaks=np.delete(peaks,1)
        # except Exception as e:
        #     pass
        # try:
        #     peaks=np.delete(peaks,2)
        # except Exception as e:
        #     pass
        # print("peaks: "+ np.array2string(peaks, separator=', '))
        print(peaks)
        for p in peaks:
            plt.plot(yval[p],histVal[p],'*r')
            # print(yval[p],histVal[p])
        # ##################    
        x,y,z,index=render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,2])
        fig,ax = plt.subplots(1)
        plt.scatter(data[index,0],data[index,1],color='blue')
        # plt.ylim(-20,)
        plt.xlim(data[index,0].min(),data[index,0].max())
        x=data[:,0]
        min_x=math.ceil(x.min())
        max_x=math.ceil(x.max())
        nbin=max_x-min_x
        x_val=np.linspace(min_x,max_x,nbin)
        arr=[]
        for p in peaks:
            arr.append(yval[p])
        # for y in arr:   
        #     lane.DisplayBins(x_val,y,'red')
        # lane_find_peaks()
        lanes =lane.DetectLanes(data[index,0:4],1,50,arr,min_x,max_x,len(peaks))
        print(lanes.shape)
        no_of_lanes_detected=lanes.shape[2]
        new_lanes=np.zeros(no_of_lanes_detected)
        # for i in range(no_of_lanes_detected):
        #     l=lanes[:,:,i]
        #     l=[l[i,:] for i in range(len(l)) if l[i,0]!=0 and l[i,1]!=0]
        #     l=np.array(l).reshape(-1,4) 
        #     plt.plot(l[:,0],l[:,1],color='yellow')
        # render_lanes_on_image(lanes,rgb, calib, w,h,filename)
        fitted_lane=fit_polynomial(lanes,peaks)
        if(len(fitted_lane)>2):
            fitted_lane = remove_shortfits(fitted_lane)
        for d in fitted_lane:
            plt.plot(d[:,0],d[:,1],color='red')
        plt.savefig('lidar_scatter/'+filename+'.png')
        render_lanes_on_image(fitted_lane,rgb, calib, w,h,filename)
        # plt.show()
        print("Done")

# /mnt/c/users/sivad/Desktop/Sensor Fusion/ADAS Lane Detection/


if __name__ == '__main__':
    main()