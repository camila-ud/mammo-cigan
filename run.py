from train import *
from config import *

if __name__ == '__main__':
    # model name to load?
    load_name = 'model_name_sou'
    # new model save name
    save_name = 'model_name_sou'

    # checkpoint number to load?
    ckpt_num = None

    # only validating the model
    if '--val' in sys.argv:
        # do not save new model
        new_model = False 
        limits = [None, None, None]
        # ckpt_num = sys.argv[2]
        # print("loading",ckpt_num)
        # instantiate GAN model
        GAN = GENGAN(save_name, load_name, patch_size, learn_rate, epochs,
                     batch_size, new_model,train_vgg=False, load_vgg=False, 
		     load_weights=False, limits=limits,use_c=True,ckpt_num=None)
             

        GAN.build_model()
        GAN.validate_model()
    elif '--syn' in sys.argv:
            # do not save new model
            new_model = False 
            limits = [None, None, None]
            test_id = sys.argv[2]
            print("sin",test_id)
            # instantiate GAN model
            GAN = GENGAN(save_name, load_name, patch_size, learn_rate, epochs,
                        batch_size, new_model,train_vgg=False, load_vgg=False, 
                load_weights=False, limits=limits,use_c=True,ckpt_num=None)
               

            GAN.build_model()
            GAN.synthesis(test_id)
    elif '--convert' in sys.argv:
            # do not save new model
            new_model = False 
            limits = [None, None, None]
            test_id = sys.argv[2]
            print("No",test_id)
            # instantiate GAN model
            GAN = GENGAN(save_name, load_name, patch_size, learn_rate, epochs,
                        batch_size, new_model,train_vgg=False, load_vgg=False, 
                load_weights=False, limits=limits,use_c=True,ckpt_num=None)
               

            GAN.build_model()
            GAN.synthesize_dataset(test_id)
    else:
        # pretrain with VGG
        train_vgg = True
        # load VGG pre-trained model
        load_vgg = True
        # load GAN model weights
        load_weights = False
        # save new model?
        new_model = True
        # sample rates for each class
        sample_rates = [1.0 / c_dims] * c_dims

        # dataset to use for limited datasets
        pos_lim = 'train_64_rand'
        limits = ['train_rand', pos_lim]

        GAN = GENGAN(save_name, load_name, patch_size, epochs,
                     batch_size, new_model, sample_rates=sample_rates,
                     ckpt_num=None, train_vgg=train_vgg, load_vgg=load_vgg,
                     load_weights=load_weights, limits=limits,use_c=True)

        GAN.build_model()
        GAN.train_model()
