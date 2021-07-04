import pickle
import masks_backup

masks_dict = masks_backup.mask_dict
with open('masks.pickle', 'wb') as fp:
    pickle.dump(masks_dict, fp)
