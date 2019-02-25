import os, sys, shutil
import itertools

from joblib import Parallel, delayed

import numpy as np
import cv2

#feature detechtion globals
DETECTION_WIDTH = 1000
SIFT = cv2.xfeatures2d.SIFT_create()

#jobs globals
TEMP_FOLDER = 'process'
JOB_COUNT = 8

#matching parameter globals
RATIO = .6
MIN_MATCH_COUNT = 10

#output globals
MAX_RESULT_WIDTH = 15000


def keyToSlice(kp):
    return np.array([kp.pt[0], kp.pt[1], kp.size,
        kp.angle, kp.response, kp.octave, kp.class_id], dtype=np.float32)
def sliceToKey(slice):
    return cv2.KeyPoint(slice[0], slice[1], slice[2],
        slice[3], slice[4], slice[5], slice[6])

#stores an image array and associated SIFT details
class Image:
    def __init__(self, filename, index):
        self.index = index
        self.data = cv2.imread(filename) #actual image array

        self.h = self.data.shape[0] #height
        self.w = self.data.shape[1] #width

    def compute(self, index):
        scale = max(self.h, self.w) / DETECTION_WIDTH
        
        adjusted = cv2.resize(self.data, (int(self.w/scale), int(self.h/scale)))

        kp, des = SIFT.detectAndCompute(adjusted, None) #SIFT keypoints and descriptors

        self.n = len(kp)

        self.kp = np.memmap(TEMP_FOLDER+'/kp_'+str(index), dtype=np.float32,
            mode = 'w+', shape= (self.n, 7))
        self.des = np.memmap(TEMP_FOLDER+'/des_'+str(index), dtype=np.float32,
            mode = 'w+', shape = (self.n, 128))
            
        for index in range(0, self.n):
            self.kp[index][:] = keyToSlice(kp[index])[:]
        self.des[:] = des[:]

        self.kp.flush()
        self.des.flush()
        
def orientation(A,B,C):
    return (B[0][1] - A[0][1])*(C[0][0] - B[0][0]) - (C[0][1] - B[0][1])*(B[0][0] - A[0][0])

def isConvex(boundary):
    A = boundary[0]
    B = boundary[1]
    C = boundary[2]
    D = boundary[3]

    if orientation(A, B, C) > 0:
        if orientation(B, C, D) > 0:
            if orientation(C, D, A) > 0:
                if orientation(D, A, B) > 0:
                    return True
    return False


def computeTransform(A, A_pts, A_size, B, B_pts, B_size):
    print(A,B,":    Matching...")
    A_kp_raw = np.memmap(TEMP_FOLDER+'/kp_'+str(A),
        dtype=np.float32,mode='r', shape=(A_pts,7))
    A_des = np.memmap(TEMP_FOLDER+'/des_'+str(A),
        dtype=np.float32,mode='r', shape=(A_pts,128))

    B_kp_raw = np.memmap(TEMP_FOLDER+'/kp_'+str(B),
        dtype=np.float32,mode='r', shape=(B_pts,7))
    B_des = np.memmap(TEMP_FOLDER+'/des_'+str(B),
        dtype=np.float32,mode='r', shape=(B_pts,128))

    result = np.memmap(TEMP_FOLDER+'/result_'+str(A)+'_'+str(B), 
        dtype=np.float32, mode='w+', shape=(3,3))
    result_inverse = np.memmap(TEMP_FOLDER+'/result_'+str(B)+'_'+str(A), 
        dtype=np.float32, mode='w+', shape=(3,3))

    #A_kp = sliceToKeyVector(A_kp_raw)
    A_kp = [sliceToKey(kp) for kp in A_kp_raw]
    B_kp = [sliceToKey(kp) for kp in B_kp_raw]
    #B_kp = sliceToKeyVector(B_kp_raw)    

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_query = flann.knnMatch(A_des, B_des, k=2)
    #matches_train = flann.knnMatch(B_des, A_des, k=2)

    good = []
    for i in range(0, len(matches_query)):
        if matches_query[i][0].distance < 0.6 * matches_query[i][1].distance: #ratio test A
            #r = matches_query[i][0].trainIdx
            #if matches_train[r][0].distance < 0.6 * matches_train[r][1].distance: #ratio test B
                #if matches_train[r][0].trainIdx == i: #symmetric test
            good.append(matches_query[i][0])

    print(A,B,":    ", len(good), "matches found")
    
    if len(good) > MIN_MATCH_COUNT:
        print(A,B,":    generating homography")
        src_pts = np.float32([ A_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ B_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        print('\n', M)

        if M is not None:
            if np.linalg.det(M) > .01:
                I = np.linalg.inv(M)

                A_boundary = np.float32([ [0,0],[0,A_size[1]],[A_size[0],A_size[1]],[A_size[0],0] ]).reshape(-1,1,2)
                B_boundary = np.float32([ [0,0],[0,B_size[1]],[B_size[0],B_size[1]],[B_size[0],0] ]).reshape(-1,1,2)

                A_warped = cv2.perspectiveTransform(A_boundary, M)
                B_warped = cv2.perspectiveTransform(B_boundary, I)

                if isConvex(A_warped):
                    if isConvex(B_warped):
                        print(A,B,":    homography is valid")
                        result[:] = M[:]
                        result_inverse[:] = I[:]
                        del result
                        del result_inverse
                        return

    print(A,B,":    homography is invalid")
    result[:] = np.zeros((3,3), dtype=np.float32)[:]
    result_inverse[:] = np.zeros((3,3), dtype=np.float32)[:]
    del result
    del result_inverse                
    return

class Box:
    def __init__(self):
        self.x_min = np.inf
        self.x_max = -np.inf
        self.y_min = np.inf
        self.y_max = -np.inf

    def addPoint(self, point):

        self.x_min = min(self.x_min,point[0])
        self.y_min = min(self.y_min,point[1])
        self.x_max = max(self.x_max,point[0])
        self.y_max = max(self.y_max,point[1])
    def addPoints(self, points):
        for point in points:
            self.addPoint(point[0])

class ImageSet:
    def __init__(self, images):
        self.data = images
        self.n = len(images) #number of images in set
        self.dist = np.full((self.n, self.n), -1, np.int32) #stores pair-wise distances
        self.dep = np.full((self.n, self.n), -1, np.int32) #stores path dependency for shortest paths
        self.transforms = np.zeros((self.n, self.n, 3,3), np.float32)
        #self.transforms = [[np.identity(3, np.float32) for _ in range(0, self.n)]  for _ in range(0, self.n)] #stores homologies between image combinations (or 0's if none is found)

        self.centers = []
        self.components = [-1 for _ in range(0, self.n)]

        self.joint_transforms = [np.identity(3, np.float32) for _ in range(0, self.n)]
        self.boundaries = [[] for _ in range(0,self.n)]
        self.boxes = [Box() for _ in range(0, self.n)]

        self.sifted = 1

    def loadTransforms(self):
        #reset the diagonal
        for A in range(0, self.n):
            self.dist[A][A] = 0
            self.dep[A][A] = A
            self.transforms[A][A][:] = np.identity(3,np.float32)[:]

        #copy results to array
        for A,B in itertools.combinations(range(0, self.n), 2):
            result = np.memmap(TEMP_FOLDER+'/result_'+str(A)+'_'+str(B), 
                dtype=np.float32, mode='r', shape=(3,3))
            result_inverse = np.memmap(TEMP_FOLDER+'/result_'+str(B)+'_'+str(A), 
                dtype=np.float32, mode='r', shape=(3,3))
            if np.count_nonzero(result) > 0:
                self.transforms[A][B][:] = result[:]
                self.transforms[B][A][:] = result_inverse[:]
                self.dist[A][B] = 1
                self.dist[B][A] = 1
                self.dep[A][B] = B
                self.dep[B][A] = A
            else:
                self.dist[A][B] = -1
                self.dist[B][A] = -1
                self.dep[A][B] = -1
                self.dep[B][A] = -1

    def computeTransforms(self):
        if self.sifted > 0:
            print("Computing Sift for images...")
            self.sifted = 0
            for i in range(0,self.n):
                print(i, "/", self.n)
                self.data[i].compute(i)

        #for every unique (order agnostic, non duplicate) combination we check connectivity and compute the homology
        pairs = list(itertools.combinations(range(0, self.n), 2))
        Parallel(n_jobs = JOB_COUNT)(delayed(computeTransform)(A, self.data[A].n, [self.data[A].w, self.data[A].h], 
            B, self.data[B].n, [self.data[B].w, self.data[B].h]) for A,B in pairs)

        self.loadTransforms()
    
    def computeDistances(self):
        #compute pair-wise distances and connected components via FW Algorithm
        for k in range(0, self.n):
            for i in range(0, self.n):
                for j in range(0, self.n):
                    if (self.dist[i][k] >= 0 and self.dist[k][j] >= 0):
                        if (self.dist[i][k] + self.dist[k][j] < self.dist[i][j]) or self.dist[i][j] == -1:
                            self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                            self.dep[i][j] = self.dep[i][k]

    def computeComponents(self):
        #find connected components and their centers
        for k in range(0, self.n):
            if self.components[k] < 0:
                self.components[k] = k #mark component membership

                #find center index of component
                center = k
                min_radius = 0
                for j in range(k, self.n): #assuming k is the center, whats is radius
                    min_radius = max(self.dist[k][j], min_radius)

                for i in range(k+1, self.n):
                    if self.dist[k][i] > 0:
                        self.components[i] = k #mark component membership

                        #find the radius of i
                        max_distance = 0
                        for j in range(k, self.n): 
                            max_distance = max(self.dist[i][j], max_distance)
                        if min_radius > max_distance: #if radius of i is less than current center...
                            min_radius = max_distance
                            center = i

                #change component marker to reflect center index
                for i in range(k, self.n):
                    if self.components[i] == k:
                        self.components[i] = center
                
                self.centers.append(center) #jot that down for later

    def computeJoints(self):
        #determine joint transforms (the transforms relative to the center of the connected component)
        #as well as boundary coordinates

        #for each center image, initialize its boundary and bounding box
        for i in self.centers:
            h = self.data[i].h
            w = self.data[i].w

            self.boundaries[i] = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
            self.boxes[i].addPoints(self.boundaries[i])
        
        #we expand from a the centers based on distance, allowing us to successively compute chain transforms
        #for a center image C and images A and B st there exists H:B->A and H':A->C then H'H:B->C
        #we then compute the boundaries and boxes for each image as we compute its transform

        for k in range(1, self.n):
            for i in range(0, self.n):
                if self.dist[i][self.components[i]] == k:
                    #compute component chain transform
                    dependent = self.dep[i][ self.components[i] ]
                    np.matmul(
                        self.joint_transforms[dependent],
                        self.transforms[i][dependent],
                        self.joint_transforms[i])

                    #compute boundary
                    h = self.data[i].h
                    w = self.data[i].w
                    pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)

                    self.boundaries[i] = cv2.perspectiveTransform(pts, self.joint_transforms[i])
                    self.boxes[self.components[i]].addPoints(self.boundaries[i])

    def mergeImages(self, output_template):

        for center in self.centers:
            #the size of the image for the current component
            current_box = self.boxes[center]

            #if the image bounds are over a certain size, cull
            current_box.x_max = min(current_box.x_max, MAX_RESULT_WIDTH/2)
            current_box.y_max = min(current_box.y_max, MAX_RESULT_WIDTH/2)
            current_box.x_min = max(current_box.x_min, -MAX_RESULT_WIDTH/2)
            current_box.y_min = max(current_box.y_min, -MAX_RESULT_WIDTH/2)

            h = np.int32(current_box.x_max - current_box.x_min)
            w = np.int32(current_box.y_max - current_box.y_min)

            #the offset transform for the image size
            Ht = np.array([[1,0,-current_box.x_min], [0,1,-current_box.y_min], [0,0,1]])

            #the image arrays
            print(w, h)
            result = np.zeros((w, h, 3), np.uint8)
            mask = np.zeros((w, h, 3), np.uint8)

            for i in range(0, self.n):
                if self.components[i] == center:
                    if self.dist[i][center] < 3:
                        #compute transfrom
                        transformed = cv2.warpPerspective(self.data[i].data, Ht.dot(self.joint_transforms[i]), (h, w))

                        #adjust the boundary for the offset
                        boundary = np.array(np.add(self.boundaries[i], [-current_box.x_min, -current_box.y_min]), np.int32)

                        #draw mask
                        mask.fill(0)
                        mask = cv2.fillPoly(mask, [boundary], (255,255,255))

                        cv2.imwrite("masks/mask_" + str(i) + ".jpg", mask)

                        #copy masked transform to result
                        result = np.where(transformed, transformed, result)

            cv2.imwrite(output_template + str(center) + ".jpg", result)

    def mergeDependents(self, output_template):
        for i in range(0, self.n):
            dep = self.dep[i][self.components[i]]

            h_target = self.data[i].h
            w_target = self.data[i].w
            h_dep = self.data[dep].h
            w_dep = self.data[dep].w
            pts_target = np.float32([ [0,0],[0,h_target],[w_target,h_target],[w_target,0] ]).reshape(-1,1,2)
            pts_dep = np.float32([ [0,0],[0,h_dep],[w_dep,h_dep],[w_dep,0] ]).reshape(-1,1,2)
            
            boundary_target = cv2.perspectiveTransform(pts_target, self.transforms[i][dep])

            current_box = Box()
            current_box.addPoints(pts_dep)
            current_box.addPoints(boundary_target)

            #if the image bounds are over a certain size, cull
            current_box.x_max = min(current_box.x_max, MAX_RESULT_WIDTH/2)
            current_box.y_max = min(current_box.y_max, MAX_RESULT_WIDTH/2)
            current_box.x_min = max(current_box.x_min, -MAX_RESULT_WIDTH/2)
            current_box.y_min = max(current_box.y_min, -MAX_RESULT_WIDTH/2)

            h = np.int32(current_box.x_max - current_box.x_min)
            w = np.int32(current_box.y_max - current_box.y_min)

            #the offset transform for the image size
            Ht = np.array([[1,0,-current_box.x_min], [0,1,-current_box.y_min], [0,0,1]])

            #the image arrays
            print(w, h)
            #result = np.zeros((w, h, 3), np.uint8)
            mask = np.zeros((w, h, 3), np.uint8)

            result = cv2.warpPerspective(self.data[dep].data, Ht.dot(self.transforms[0][0]), (h, w))
            transformed_target = cv2.warpPerspective(self.data[i].data, Ht.dot(self.transforms[i][dep]), (h, w))

            #adjust the boundary for the offset
            boundary = np.array(np.add(boundary_target, [-current_box.x_min, -current_box.y_min]), np.int32)

            #draw mask
            mask.fill(0)
            mask = cv2.fillPoly(mask, [boundary], (255,255,255))

            cv2.imwrite("masks/mask_" + str(i) + ".jpg", mask)

            #copy masked transform to result
            result = np.where(mask, transformed_target, result)

            cv2.imwrite(output_template + str(i) + ".jpg", result)
#from a set of images, computes the homological connectivity graph
def merge_images(input_folder, output_template):

    print("Loading files...")
    group = load_directory(input_folder)

    print("Computing transforms...")
    print("members: ", group.n)
    group.computeTransforms()

    print("Computing distances...")
    group.computeDistances()

    print("Computing Components and Centers")
    group.computeComponents()

    print("Computing joint transforms...")
    group.computeJoints()

    print("merging images...")
    group.mergeImages(output_template)

    print("Done!")


def load_directory(path):
    if path[-1] != '/':
        path += '/'
    directory = os.listdir( path )
    n = len(directory)
    i = 1
    images = []
    print(os.getcwd())
    if not os.path.exists(TEMP_FOLDER):
        #shutil.rmtree(TEMP_FOLDER)
        os.mkdir(TEMP_FOLDER)
    for index, file_name in enumerate(directory):
        print(i, "/", n)
        i+=1
        images.append(Image(path + file_name, index))
    
    return ImageSet(images)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Please provide and input folder and output template.")
        print('if "result_0.jpg" is desired, enter "result_" as template.')
    else:
        merge_images(sys.argv[1], sys.argv[2])
