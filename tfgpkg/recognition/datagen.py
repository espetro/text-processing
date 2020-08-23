from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame

class DataGen:
    """Wrapper class for Keras' ImageDataGenerator. Its logic has a hard dependency on what DataUnpack outputs
    
    Parameters
    ----------
        train: Either[Tuple[ndarray, ndarray, ndarray], DataFrame]
            A train split containing images.

            The direct output from `DataUnpack.unpack`, which is either a tuple of np.ndarray (image, label, vector)
            that holds a memory-based image, or a `pandas.DataFrame` that holds the path to a disk-based image.

        test: Either[Tuple[ndarray, ndarray, ndarray], DataFrame]
            A test split containing images.
            
        valid: Either[Tuple[ndarray, ndarray, ndarray], DataFrame]
            A validation split containing images.

        train_options: dict
            A dict holding options for the ImageDataGenerator object used for the training set (zoom_range, etc.)ñ
    """

    def __init__(self, train, test, valid, train_options):
        if isinstance(train, DataFrame) or isinstance(train, tuple):
            self.train = train
            self.train_gen = ImageDataGenerator(rescale=1/255., **train_options)

            self.test = test
            self.test_gen = ImageDataGenerator(rescale=1/255.)

            self.valid = valid
            self.valid_gen = ImageDataGenerator(rescale=1/255.)
        else:
            raise ValueError("Expected either a pandas DataFrame or a tuple of numpy.ndarray")

    def from_df(self, dirs, common_options):
        """
        Parameters
        ----------
            dirs: Tuple[str]
                A tuple containing output folders for (train, test, validation) splits
            
            common_options: dict
                A dictionary holding common options for the three data generators (target_size, batch_size, mode, etc.)
        """
        train_dir, test_dir, valid_dir = dirs

        if isinstance(self.train, DataFrame):
            train_flow = self.train_gen.flow_from_dataframe(self.train, train_dir, **common_options)
            test_flow = self.test_gen.flow_from_dataframe(self.test, test_dir, **common_options)
            valid_flow = self.valid_gen.flow_from_dataframe(self.valid, valid_dir, **common_options)

            return train_flow, test_flow, valid_flow
        else:
            raise ValueError(f"Expected a DataFrame but got {type(self.train)}")

    def from_tuple(self, batch):
        """
        Parameters
        ----------
            batch: int
                The batch size to use
        """
        if isinstance(self.train, tuple):
            self.train_gen.fit(self.train[0])
            self.test_gen.fit(self.test[0])
            self.valid_gen.fit(self.valid[0])

            train_flow = self.train_gen.flow(self.train[0], self.train[-1], batch_size=batch)
            test_flow = self.test_gen.flow(self.test[0], self.test[-1], batch_size=batch)
            valid_flow = self.valid_gen.flow(self.valid[0], self.valid[-1], batch_size=batch)

            return train_flow, test_flow, valid_flow
        else:
            raise ValueError(f"Expected a Tuple of numpy ndarrays but got {type(self.train)}")


if __name__ == "__main__":
    pass
    # train_opts = dict(width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2)
    # batch = 128
    #
    # train, test, valid = DataUnpack.unpack(..., save_to_disk=False)  # returns a tuple of ndarrays
    # 
    # train_gen, test_gen, valid_gen = DataGen(train, test, valid, **train_opts).from_tuple(batch)