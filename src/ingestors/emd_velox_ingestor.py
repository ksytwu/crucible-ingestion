import os
import re
import io

from pathlib import Path
from PIL import Image
import numpy as np
import ncempy.io as nio
import matplotlib.pyplot as plt
import logging
import json

from .crucible_ingestor import CrucibleDatasetIngestor
from crucible import CrucibleClient
from crucible.models import Dataset
from ..utils import get_secret


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Crucible Client
crucible_api_url = os.environ.get('CRUCIBLE_API_URL')
apikey = get_secret("CRUCIBLE_APIKEY", "crucible_admin_apikey/versions/4")
client = CrucibleClient(api_url=crucible_api_url, api_key=apikey)

def parse_dataset_as_dict(dataset): 
    """
    Turns a h5py Dataset of type "O" into a python dict.
    Helper function for processing certain data from the h5py file. 
    Modeled after rsciio.

    Args:
    - dataset: an h5py Dataset object of type "O"

    Returns: 
    - dataset decoded as a dict 
    """
    bytes = dataset[()].tolist()[0]
    decoded = bytes.decode("utf-8", "ignore").rstrip("\x00")
    return json.loads(decoded)

get_groupType_from_md = lambda md: md['General']['groupType'] if 'groupType' in md['General'] else "" # options: STEM, TEM, EDS_Processed, EDS
get_title_from_md = lambda md: md['General']['title'] if 'title' in  md['General'] else "" # options: EDS_SI, EDS_Spectrum, HAADF, element name

PROCESSED_IMAGE_GROUP_NAME = 'EDS_Processed'
SPECTRUM_IMAGE_GROUP_NAME = 'EDS_SI'
SPECTRUM_GROUP_NAME = 'EDS_Spectrum'


class fileEMDVeloxWithSpectra(nio.emdVelox.fileEMDVelox):
    def __init__(self, filename): 
        super().__init__(filename)
        self._parse_image_titles()
    
    def _find_groups(self):
        """ 
        Find all data groups: spectrum data, image data, and spectrum image. (in that order)
        Spectrum are omitted unless there are no other data types.
        """
        self.list_data = []
        groups_to_extract = ['Data/Image', 'Data/SpectrumImage']
        for group in groups_to_extract: 
            if group in self._file_hdl:
                self.list_data += list(self._file_hdl[group].values())
        
        # only add Spectrum data (to the front of list_data) if there are no Images or Spectrum Images 
        if len(self.list_data) == 0 and 'Data/Spectrum' in self._file_hdl: 
            self.list_data = list(self._file_hdl['Data/Spectrum'].values()) + self.list_data
    
    def _parse_image_titles(self):
        """
        Create a mapping between each image_uuid: 
            - 'groupType' (EDS, STEM, TEM)
            - 'title' (element name or HAADF)
        Stores the mapping for in self.img_titles
        """
        self.img_titles = {}
        if "Displays/ImageDisplay" not in self._file_hdl: 
            return 
        list_of_datasets = self._file_hdl["Displays/ImageDisplay"]
        for group in list_of_datasets.values():
            dataset_dict = parse_dataset_as_dict(group)
            if "data" not in dataset_dict: # ignore any images that are not linked to data
                continue
            else:
                data_ref_path = dataset_dict["data"]
                data_ref_dict = parse_dataset_as_dict(self._file_hdl.get(data_ref_path))
                data_path = data_ref_dict["dataPath"]
                stem_detectors = ['BF', 'DF']
                if any(detectors in dataset_dict['title'] for detectors in stem_detectors): #if DF or BF in title
                    groupType = "STEM"
                else:
                    groupType = PROCESSED_IMAGE_GROUP_NAME
                self.img_titles[data_path] = {'groupType': groupType, 
                                          'title': dataset_dict['title']} 

    def getMetadata(self, group):
        """ Reads important metadata from Velox EMD files.

        Parameters
        ----------
        group : h5py.Group or int
            The h5py group to load the metadata from which is easily retrived from the list_data attribute.
            If input is an int then the group corresponding to list_data attribute is used. The string 
            metadata is loaded and parsed by the json module into a dictionary.
        """
        # ensure group is an actual group object, not an int 
        try:
            if type(group) is int:
                group = self.list_data[group]
        except IndexError:
            raise IndexError('EMDVelox group #{} does not exist.'.format(group))
        
        meta_data = super().getMetadata(group)
       
        # add general metadata
        general_md = {'groupType': "", 'title': ""} # placeholder 
        if group.name in self.img_titles: 
            general_md = self.img_titles[group.name]
        elif group.parent.name == '/Data/Spectrum':
            general_md = {'groupType': SPECTRUM_GROUP_NAME, 'title': SPECTRUM_GROUP_NAME} 
        elif group.parent.name == '/Data/SpectrumImage':
            general_md = {'groupType': SPECTRUM_IMAGE_GROUP_NAME, 'title': SPECTRUM_IMAGE_GROUP_NAME} 
        else: # default for images whose titles aren't included in Display Group? 
            general_md = {'groupType': 'Image', 'title': 'Image'} 

        meta_data.update({'General': general_md})

        return meta_data


    def getThumbnail(self):
        """
        Returns the saved thumbnail image from the EMD file, if it exists.
        """
        if 'Thumbnail.jpg' not in self._file_hdl:
            logger.info(f'Thumbnail.jpg not in file')
            return None
        thumbnail_dataset = self._file_hdl['Thumbnail.jpg']
        thumbnail_image = Image.open(io.BytesIO(thumbnail_dataset[()].tobytes()))
        return thumbnail_image

    def getThumbnailImageDataset(self):
        """
        Return an image dataset for thumbnail-generation. 
        If a non-processed-image exists, return an arbitrary one. 
        Otherwise, return an arbitrary processed-image. 
        """
        # print(f"Got image_array for thumbnail: {len(image_dataset) if image_dataset!=None else 'None'}")
        if '/Data/Image' not in self._file_hdl: 
            logger.info(f'/Data/Image not in file')
            return None

        # choose the image from which to create a thumbnail
        image_dataset = None
        # iterate through image datasets to find a non-processed image 
        all_image_groups = list(self._file_hdl['/Data/Image'].values())
        for group in all_image_groups[::-1]:  # start from the back, assuming that processed images are at the front
            md = self.getMetadata(group)
            if get_groupType_from_md(md) != PROCESSED_IMAGE_GROUP_NAME: 
                image_dataset = group 
                break 
        # if all images are processed images, use an arbitrary processed image
        if image_dataset is None: 
            image_dataset = all_image_groups[0]
        
        return image_dataset['Data'][:,:,0]
        
class VeloxEmdIngestor(CrucibleDatasetIngestor):
    '''subclass for ingesting Velox EMD files'''

    def is_file_supported(self):
        if not self.file_to_upload.endswith('.emd'):
            return False
        
        with fileEMDVeloxWithSpectra(self.file_to_upload) as emd1:
            return len(emd1.list_data) > 0
            # this only ensures that there's at least a Image, Spectrum Image, or Spectrum group? 

        #TODO: modify

    
    def get_scientific_metadata(self):
        """
        Updates scientific metadata and measurement. 
        """
        self.scientific_metadata, self.measurement = self._parse_measurement_metadata()
        # logger.info(f'Got metadata from Velox EMD: {self.scientific_metadata=}')

    def get_dataset_metadata(self):
         # Use parent class method to set data_format, size, and source_folder
        CrucibleDatasetIngestor.get_dataset_metadata(self)
        
    def parse_dataset_name(self):
        if self.dataset_name:
            return
        else:
            # no_ext_fname = os.path.splitext(self.file_to_upload)[0]
            self.dataset_name = os.path.basename(self.file_to_upload) # keep the file extension
            logger.info(f"{self.dataset_name=}")
            return

    def _generate_thumbnail_from_array(self, image_array): 
        """Generate a thumbnail from an image_arr. Helper function. 

        Returns
        -------
        : PIL.Image
            Thumbnail image as a PIL Image object.
        """
        target_size = (200, 200) # pixels
        dpi = 100
        fig_size = (target_size[0] / dpi, target_size[1] / dpi) # inches

        fg, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
        ax.imshow(image_array, cmap = 'gray')
        ax.axis('off')

        # Convert to PIL Image and store in self.thumbnails
        buf = io.BytesIO()
        fg.savefig(buf, bbox_inches='tight', pad_inches=0.05, dpi=100)
        im = Image.open(buf)
        return im

    def generate_thumbnail(self):
        """Generate a thumbnail from an EMD image as a PNG.

        Returns
        -------
        : PIL.Image
            Thumbnail image as a PIL Image object.
        """       
        try:            
            with fileEMDVeloxWithSpectra(self.file_to_upload) as emd1:
                thumbnail_img = emd1.getThumbnail()
                if thumbnail_img is not None:
                    return thumbnail_img
                else:
                    logger.info(f"No thumbnail found.")
            return None
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")

    def get_thumbnails(self):
        try:
            thumbnail = self.generate_thumbnail()
            if thumbnail:
                self.add_thumbnail(thumbnail, "Velox_EMD_Thumbnail")
            logger.info(f'{len(self.thumbnails)=}')
        except Exception as e:
            logger.error(f"Failed to extract thumbnail: {e}")


    def _parse_measurement_metadata(self): 
        """
        Parses scientific metadata for each child/measurement, returns list of metadata dictionaries. 
        """
        illumination, projection, signal = "", "", ""

        all_metadata = {}
        with fileEMDVeloxWithSpectra(self.file_to_upload) as emd1:
            first_md = emd1.getMetadata(0)
            illumination = self.get_illumination_mode(first_md)
            projection = self.get_projection_mode(first_md)
            signal = ""

            for i, group in enumerate(emd1.list_data): 
                md = emd1.getMetadata(group)

                measurement_type = group.parent.name[6:]
                # parse measurement_type for this measurement
                # illumination = self.get_illumination_mode(md)
                # projection = self.get_projection_mode(md)
                cur_signal = self.get_signal_type(md) if measurement_type != 'SpectrumImage' else 'EDS' # handle spectrum images separately? 
                measurement_str = illumination + ' ' + projection + ' ' + cur_signal if cur_signal else 'Velox Processed Image'
                md.update({"measurement": measurement_str})

                # update overall measurement
                if signal == "" and cur_signal != "": 
                    signal = cur_signal
                elif signal != "EDS": # Priority: EDS > Pixelated > Non-Pixelated
                    if cur_signal == "EDS": # as soon as there's an EDS, signal should be EDS
                        signal = "EDS"
                    if signal != "Pixelated":
                        if cur_signal == "Pixelated":
                            signal = "Pixelated" # bump up signal to Pixelated   

                # generate thumbnail for image measurements (only for children ds; otherwise, parent ds's thumbnail handled separately later)
                # note: thumbnail will be added to crucible, but encoding will be removed from md dictionary later
                if len(emd1.list_data) > 1: 
                    if measurement_type == "Image":
                        image_array = group['Data'][:,:,0]
                        thumbnail = self._generate_thumbnail_from_array(image_array)
                        md.update({"thumbnail": thumbnail})
                    if measurement_type == "SpectrumImage": 
                        thumbnail = self.generate_thumbnail() # SpectrumImage will have same thumbnail as overall file 
                        md.update({"thumbnail": thumbnail})
                
                all_metadata.update({get_title_from_md(md): md}) # use title as key for child scientific metadata 
                # previously also included dataset path: get_title_from_md(md) + f" ({group.name})": md}

        overall_measurment_str = illumination + ' ' + projection + ' ' + signal # assume that overall measurement is never Velox Processed Image
        return all_metadata, overall_measurment_str

    def parse_children(self):
        def upload_child(md, parent_dsid): 
            """
            Uploads child dataset using metadata MD. Links the created dataset record to PARENT_DSID.
            Also, adds the thumbnail for the child, if applicable. 
            Returns child_dsid. 
            """
            # remove thumbnail from metadata (before dataset creation) if applicable 
            child_thumbnail = None
            if "thumbnail" in md: # generated in _parse_measurement_metadata
                child_thumbnail = md["thumbnail"]
                del md["thumbnail"]

            # Create dataset
            child_ds_name = self.dataset_name + f" ({get_title_from_md(md)})"
            child_ds = Dataset(
                # unique_id      = self.mfid, # need a new id for each measurement ds?
                measurement    = md['measurement'], # prev: get_groupType_from_md(md), 
                project_id     = self.project_id,
                owner_orcid    = None,  # API key handles user authentication
                dataset_name   = child_ds_name,
                # session_name   = self.session_name,
                # public         = self.public,
                # instrument_name = self.instrument_name, # TODO: update instrument
                data_format    = self.data_format,
                source_folder  = self.source_folder,
                # file_to_upload = self.files_to_upload[0] <- INCLUDE if upload_file
            )
        
            resp = client.datasets.create(
                child_ds,
                scientific_metadata=md,
                keywords=self.keywords,
            )
            child_dsid = resp['created_record']['unique_id']

            # add thumbnail for child if applicable
            if child_thumbnail is not None:
                resp = client.datasets.add_thumbnail(child_dsid, child_thumbnail, 'Velox_EMD_Thumbnail')
                logger.info(f'{child_dsid} -- thumbnail -- {resp}')
            else: 
                logger.info(f'{child_dsid} thumbnail was None')
            # Link child with parent dataset 
            client.datasets.link_parent_child(parent_dsid, child_dsid)

            return child_dsid
        
        def find_dsid_for_ds_name(ds_name, records):
            """ Helper function. 
            Returns dsid corresponding to DS_NAME, or None if DS_NAME not in RECORDS.
            """
            for r in records:
                if r['dataset_name'] == ds_name:
                    return r['unique_id']
            return None
        
        ### if there's only 1 measurement, don't need to create an additional child dataset. 
        if len(self.scientific_metadata) == 1:
            self.scientific_metadata = list(self.scientific_metadata.values())[0] # decapsulate scientific metadata for parent
            return 
        
        ### handle multi-dataset files

        # create parent_child_map for caching client.list_children() results 
        parent_child_map = {self.unique_id: client.datasets.list_children(self.unique_id)} # self.unique_id = file_dsid
        spectrum_image_dsid = None

        # upload children, ensuring Processed Images are nested under SpectrumImage; otherwise, nested under File
        for i, md in enumerate(list(self.scientific_metadata.values())[::-1]): # assume SpectrumImage is at the end of sci_metadata, so iterate backwards
            # determine if parent is file or SpectrumImage
            parent_dsid = spectrum_image_dsid if (get_groupType_from_md(md) == PROCESSED_IMAGE_GROUP_NAME and spectrum_image_dsid != None) else self.unique_id 
            
            # get parent's existing_children: 
            existing_children = []
            if parent_dsid in parent_child_map: 
                existing_children = parent_child_map[parent_dsid]
            else: 
                existing_children = client.datasets.list_children(parent_dsid)
                parent_child_map[parent_dsid] = existing_children # update parent_child_map

            child_ds_name = f"{self.dataset_name} ({get_title_from_md(md)})"
            found_dsid = find_dsid_for_ds_name(child_ds_name, existing_children)
            if found_dsid is not None: # child already exists 
                # don't upload this child -- but keep track of si_dsid if it's a spectrum image
                if i == 0 and get_groupType_from_md(md) == SPECTRUM_IMAGE_GROUP_NAME:
                    spectrum_image_dsid = found_dsid # keep track of si_dsid 
                logger.info(f"skip upload for {child_ds_name}")
            else: 
                # child doesn't exist yet; upload as normal
                dsid = upload_child(md, parent_dsid)
                
                if i == 0 and get_groupType_from_md(md) == SPECTRUM_IMAGE_GROUP_NAME:
                    spectrum_image_dsid = dsid # keep track of si_dsid 
                logger.info(f"uploaded {child_ds_name} with dsid {dsid}")


    def get_illumination_mode(self, metadata_dictionary): 
        """
        Identify the illumination mode for the measurement/child dataset 
        corresponding to to METADATA_DICTIONARY.

        Currently, Spectre specific logic. 

        Args: 
        - metadata_dictionary: dict 

        Returns: 
        - illumination: str (defaults to '[IlluminationMode]' no case matched)
        """
        # handle errors/edge cases
        illumination = '[IlluminationMode]' # placeholder 
        if 'OperatingMode' not in metadata_dictionary: 
            return illumination
        
        data_illumination = int(metadata_dictionary['OperatingMode'])
        if data_illumination == 1:
            illumination = 'TEM'
        elif data_illumination == 2:
            illumination = 'STEM'
        return illumination

    def get_projection_mode(self, metadata_dictionary): 
        """
        Identify the projection mode for the measurement/child dataset 
        corresponding to to METADATA_DICTIONARY.

        Currently, Spectre specific logic. 

        Args: 
        - metadata_dictionary: dict 

        Returns: 
        - projection: str (defaults to '[ProjectorMode]' no case matched) 
        """
        # handle errors/edge cases
        projection = '[ProjectorMode]' # placeholder 
        if 'ProjectorMode' not in metadata_dictionary: 
            return projection

        data_proj = int(metadata_dictionary['ProjectorMode'])
        if data_proj == 1:
            projection = 'Diffraction'
        elif data_proj == 2:
            projection = 'Imaging'
        return projection

    def get_signal_type(self, metadata_dictionary):
        """
        Identify the projection mode for the measurement/child dataset 
        corresponding to to METADATA_DICTIONARY.

        Currently, Spectre specific logic. 

        Args: 
        - metadata_dictionary: dict 

        Returns: 
        - signal: str or None (defaults to None if no case matched) 
        """
        # handle errors/edge cases
        signal = "" 
        if 'DetectorIndex' not in metadata_dictionary: 
            return signal # want measurement = 'Velox Processed Image'
        
        detector_indx = int(metadata_dictionary['DetectorIndex'])
        if detector_indx in [1, 3, 4, 5]:
            signal = 'Pixelated'
        elif detector_indx in [0, 2, 6]:
            signal = 'Non-Pixelated'
        elif detector_indx in [7, 8, 9, 10, 11, 12]:
            signal = 'EDS'
        return signal