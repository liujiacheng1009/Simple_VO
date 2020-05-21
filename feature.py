
import cv2 

kNumFeatureDefault = 2000
kLkPyrOpticFlowNumLevelsMin = 3   # maximal pyramid level number for LK optic flow 

class FeatureTrackingResult(object): 
    def __init__(self):
        self.kps_ref = None          # all reference keypoints (numpy array Nx2)
        self.kps_cur = None          # all current keypoints   (numpy array Nx2)
        self.des_cur = None          # all current descriptors (numpy array NxD)
        self.idxs_ref = None         # indexes of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs_cur = None         # indexes of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)
        self.kps_ref_matched = None  # reference matched keypoints, kps_ref_matched = kps_ref[idxs_ref]
        self.kps_cur_matched = None  # current matched keypoints, kps_cur_matched = kps_cur[idxs_cur]

class ShiTomasiDetector(object): 
    def __init__(self, num_features=kNumFeatureDefault, quality_level = 0.01, min_coner_distance = 3):
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_coner_distance = min_coner_distance
        self.blockSize=5    # 3 is the default block size 
        # we use LK pyr optic flow for matching     
        self.lk_params = dict(winSize  = (21, 21), 
                              maxLevel = max(kLkPyrOpticFlowNumLevelsMin,3),
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))  


    def detectAndCompute(self, frame, mask=None):
        pts = cv2.goodFeaturesToTrack(frame, self.num_features, self.quality_level, self.min_coner_distance, blockSize=self.blockSize, mask=mask)
        # convert matrix of pts into list of keypoints 
        if pts is not None: 
            kps = [ cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts ]
        else:
            kps = [] 
        if len(kps) > self.num_features:
            kps = sorted(kps, key=lambda x:x.response, reverse=True)[:num_features]  
        return kps , None     

        # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref = None):
        kps_cur, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, kps_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        res = FeatureTrackingResult()    
        #res.idxs_ref = (st == 1)
        res.idxs_ref = [i for i,v in enumerate(st) if v== 1]
        res.idxs_cur = res.idxs_ref.copy()       
        res.kps_ref_matched = kps_ref[res.idxs_ref] 
        res.kps_cur_matched = kps_cur[res.idxs_cur]  
        res.kps_ref = res.kps_ref_matched  # with LK we follow feature trails hence we can forget unmatched features 
        res.kps_cur = res.kps_cur_matched
        res.des_cur = None                      
        return res  