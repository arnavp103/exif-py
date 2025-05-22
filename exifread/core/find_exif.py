"""Utilities to find the EXIF offset and endian."""

import struct
from typing import BinaryIO, Dict, Tuple, Optional

from exifread.core.exceptions import ExifNotFound, InvalidExif
from exifread.core.heic import HEICExifFinder, find_heic_tiff
from exifread.core.jpeg import find_jpeg_exif
from exifread.core.utils import ord_
from exifread.exif_log import get_logger

logger = get_logger()


ENDIAN_TYPES: Dict[str, str] = {
    "I": "Intel",
    "M": "Motorola",
    "\x01": "Adobe Ducky",
    "b": "XMP/Adobe unknown",
}


def get_endian_str(endian_bytes) -> Tuple[str, str]:
    endian_str = chr(ord_(endian_bytes[0]))
    return endian_str, ENDIAN_TYPES.get(endian_str, "Unknown")


def find_tiff_exif(fh: BinaryIO) -> Tuple[int, bytes]:
    logger.debug("TIFF format recognized in data[0:2]")
    fh.seek(0)
    endian = fh.read(1)
    fh.read(1)
    offset = 0
    return offset, endian


def find_webp_exif(fh: BinaryIO) -> Tuple[int, bytes]:
    logger.debug("WebP format recognized in data[0:4], data[8:12]")
    # file specification: https://developers.google.com/speed/webp/docs/riff_container
    data = fh.read(5)
    if data[0:4] == b"VP8X" and data[4] & 8:
        # https://developers.google.com/speed/webp/docs/riff_container#extended_file_format
        fh.seek(13, 1)
        while True:
            data = fh.read(8)  # Chunk FourCC (32 bits) and Chunk Size (32 bits)
            if len(data) != 8:
                raise InvalidExif("Invalid webp file chunk header.")
            if data[0:4] == b"EXIF":
                fh.seek(6, 1)
                offset = fh.tell()
                endian = fh.read(1)
                return offset, endian
            size = struct.unpack("<L", data[4:8])[0]
            fh.seek(size, 1)
    raise ExifNotFound("Webp file does not have exif data.")


def find_png_exif(fh: BinaryIO, data: bytes) -> Tuple[int, bytes]:
    logger.debug("PNG format recognized in data[0:8]=%s", data[:8].hex())
    fh.seek(8)

    while True:
        data = fh.read(8)
        chunk = data[4:8]
        logger.debug("PNG found chunk %s", chunk.decode("ascii"))

        if chunk in (b"", b"IEND"):
            break
        if chunk == b"eXIf":
            offset = fh.tell()
            return offset, fh.read(1)

        chunk_size = int.from_bytes(data[:4], "big")
        fh.seek(fh.tell() + chunk_size + 4)

    raise ExifNotFound("PNG file does not have exif data.")


# Helper function to find an ISOBMFF box
def _find_box(fh: BinaryIO, box_type: bytes, search_limit: int, target_uuid: Optional[bytes] = None) -> Tuple[int, int]:
    """
    Finds an ISOBMFF box of a given type within a given search limit.
    Returns the start offset of the box's payload and the size of the box's payload.
    The search_limit is the absolute offset in the file where searching should stop.
    If target_uuid is provided (when box_type is b'uuid'), it also matches the UUID value.
    """
    current_pos = fh.tell()
    while current_pos < search_limit:
        fh.seek(current_pos)
        header = fh.read(8)
        if len(header) < 8:
            logger.debug(f"_find_box: Reached EOF or incomplete header at {current_pos}")
            break

        try:
            size = struct.unpack(">I", header[0:4])[0]
            b_type = header[4:8]
        except struct.error:
            logger.debug(f"_find_box: Could not unpack header at {current_pos}")
            break # Corrupted box

        box_header_size = 8
        box_payload_offset = current_pos + box_header_size
        actual_box_total_size = size

        if size == 1:  # Extended size (64-bit)
            extended_size_bytes = fh.read(8)
            if len(extended_size_bytes) < 8:
                logger.debug(f"_find_box: Incomplete extended size for {b_type.decode('ascii', 'replace')} at {current_pos}")
                break
            try:
                actual_box_total_size = struct.unpack(">Q", extended_size_bytes)[0]
            except struct.error:
                logger.debug(f"_find_box: Could not unpack extended size for {b_type.decode('ascii', 'replace')} at {current_pos}")
                break
            box_header_size += 8 # for the extended size field itself
        elif size == 0:  # Extends to end of file (or parent box limit)
            actual_box_total_size = search_limit - current_pos
            if actual_box_total_size < box_header_size : # Not enough space even for header
                 logger.debug(f"_find_box: Box {b_type.decode('ascii', 'replace')} size 0 but extends beyond limit or invalid.")
                 break

        if actual_box_total_size < box_header_size:
            logger.debug(f"_find_box: Invalid box size {size} (total {actual_box_total_size}) for {b_type.decode('ascii', 'replace')} at {current_pos}. Min header size is {box_header_size}.")
            break

        box_payload_size = actual_box_total_size - box_header_size
        if box_payload_size < 0:
            logger.debug(f"_find_box: Negative payload size for {b_type.decode('ascii', 'replace')} at {current_pos}")
            break

        if b_type == box_type:
            if box_type == b'uuid':
                if target_uuid is None:
                     logger.warning("_find_box: target_uuid is None for uuid box type during search.")
                     # This case should ideally not be hit if called correctly,
                     # but as a fallback, we might treat it as a generic uuid find.
                     # However, for CR3, we always expect a target_uuid.
                     # For now, let's assume this means "any uuid", which is not what we want for CR3.
                     # So we'll skip if target_uuid is not provided.
                     current_pos += actual_box_total_size
                     continue


                if box_payload_size < 16: # Not enough space for the UUID itself
                    logger.debug(f"_find_box: uuid box at {current_pos} is too small to contain a UUID.")
                    current_pos += actual_box_total_size
                    continue

                fh.seek(box_payload_offset) # Go to start of payload to read UUID
                uuid_value = fh.read(16)
                if len(uuid_value) < 16:
                    logger.debug(f"_find_box: Could not read 16 bytes for UUID in {b_type.decode('ascii', 'replace')} at {box_payload_offset}")
                    break # Cannot read UUID, stop

                if uuid_value == target_uuid:
                    # Return offset *after* UUID, and size *excluding* UUID
                    return box_payload_offset + 16, box_payload_size - 16
                # Not the UUID we are looking for, continue search
            else: # Not a UUID box type, or it is 'uuid' but we are not matching the value (target_uuid is None)
                return box_payload_offset, box_payload_size

        current_pos += actual_box_total_size
        if actual_box_total_size == 0 and size == 0 :
            logger.debug(f"_find_box: Box {b_type.decode('ascii', 'replace')} extends to end of limit, stopping search.")
            break

    raise ExifNotFound(f"Box '{box_type.decode('ascii', 'replace')}'" + \
                       (f" with UUID {target_uuid.hex()}" if target_uuid and box_type == b'uuid' else "") + \
                       f" not found within limit {search_limit}.")


def find_cr3_exif(fh: BinaryIO) -> Tuple[int, bytes]:
    """Extracts EXIF data location from CR3 files."""
    fh.seek(0)
    # Determine file size to use as the initial search limit for 'moov'
    original_pos = fh.tell()
    fh.seek(0, 2) # Go to end of file
    file_limit = fh.tell()
    fh.seek(original_pos) # Reset to original position (usually 0)

    try:
        # 1. Find 'moov' box (can be anywhere in the file)
        logger.debug("CR3: Searching for 'moov' box...")
        moov_payload_offset, moov_payload_size = _find_box(fh, b'moov', file_limit)
        logger.debug(f"CR3: Found 'moov' box payload at {moov_payload_offset}, size {moov_payload_size}")

        # 2. Find specific 'uuid' box within 'moov'
        # The UUID is b'\x85\xc0\xb6\x87\x82\x0f\x11\xe0\x81\x11\xf4\xce\x46\x2b\x6a\x48'
        target_cr3_uuid = b'\x85\xc0\xb6\x87\x82\x0f\x11\xe0\x81\x11\xf4\xce\x46\x2b\x6a\x48'
        fh.seek(moov_payload_offset) # Start searching from the beginning of moov's payload
        uuid_search_limit = moov_payload_offset + moov_payload_size
        logger.debug(f"CR3: Searching for 'uuid' box with UUID {target_cr3_uuid.hex()} within 'moov' (limit {uuid_search_limit})...")
        uuid_payload_offset, uuid_payload_size = _find_box(fh, b'uuid', uuid_search_limit, target_uuid=target_cr3_uuid)
        logger.debug(f"CR3: Found specific 'uuid' box payload at {uuid_payload_offset}, size {uuid_payload_size}")

        # 3. Attempt to find EXIF data in CMT3, CMT4, CMT2, then CMT1.
        #    Prioritizing preview image IFDs (CMT3, CMT4) then Makernotes (CMT2, CMT1)
        #    to see if the test file VZG_0005.CR3 has its main EXIF in these.
        cmt_box_types_to_try = [b'CMT3', b'CMT4', b'CMT2', b'CMT1']
        
        last_exception = None
        for cmt_type in cmt_box_types_to_try:
            try:
                fh.seek(uuid_payload_offset) # Reset seek to start of UUID payload for each attempt
                cmt_search_limit = uuid_payload_offset + uuid_payload_size
                logger.debug(f"CR3: Searching for '{cmt_type.decode()}' box within 'uuid' (limit {cmt_search_limit})...")
                
                # _find_box returns payload_offset and payload_size of the cmt_type box
                cmt_payload_offset, _ = _find_box(fh, cmt_type, cmt_search_limit)
                logger.debug(f"CR3: Found '{cmt_type.decode()}' box payload at {cmt_payload_offset}")

                # The payload of the CMT box is the TIFF data.
                tiff_offset = cmt_payload_offset
                fh.seek(tiff_offset)
                endian_bytes = fh.read(2)

                if len(endian_bytes) < 2:
                    logger.warning(f"CR3/{cmt_type.decode()}: Could not read TIFF endian marker, EOF reached at offset {tiff_offset}.")
                    # Try next CMT type
                    last_exception = InvalidExif(f"CR3/{cmt_type.decode()}: Could not read TIFF endian marker, EOF reached.")
                    continue 
                
                if endian_bytes not in [b'II', b'MM']:
                    logger.warning(f"CR3/{cmt_type.decode()}: Invalid TIFF endian marker {endian_bytes!r} at offset {tiff_offset}.")
                    # Try next CMT type
                    last_exception = InvalidExif(f"Invalid TIFF endian marker: {endian_bytes!r} in CR3/{cmt_type.decode()} at offset {tiff_offset}")
                    continue

                logger.info(f"CR3: Using '{cmt_type.decode()}' box for EXIF data at offset {tiff_offset}, endian {endian_bytes.decode('ascii')}.")
                return tiff_offset, endian_bytes[0:1] # Return offset and the first byte of endianness marker
            
            except ExifNotFound as e:
                logger.debug(f"CR3: '{cmt_type.decode()}' box not found or not suitable: {e}")
                last_exception = e
                # Continue to the next CMT type in the list
            except InvalidExif as e: # Catch InvalidExif from _find_box or our checks
                logger.debug(f"CR3: Invalid EXIF structure encountered while trying '{cmt_type.decode()}': {e}")
                last_exception = e
                # Continue to the next CMT type
        
        # If loop finishes, no suitable CMT box was found
        logger.error(f"CR3: None of the attempted CMT boxes ({[t.decode() for t in cmt_box_types_to_try]}) yielded valid EXIF data.")
        if last_exception:
            if isinstance(last_exception, ExifNotFound):
                 raise ExifNotFound(f"CR3: None of the suitable CMT boxes found. Last error: {last_exception}")
            else: # Should be InvalidExif
                 raise InvalidExif(f"CR3: None of the suitable CMT boxes yielded valid EXIF. Last error: {last_exception}")
        else: # Should not happen if list is not empty
            raise ExifNotFound("CR3: No CMT boxes found and no specific error recorded.")

    except ExifNotFound as e: # This will catch errors from finding 'moov' or 'uuid'
        logger.debug(f"CR3: Could not find required structural box: {e}")
        raise ExifNotFound(f"Required structural box not found in CR3: {e}")
    except struct.error as e:
        logger.error(f"CR3: Struct parsing error: {e}")
        raise InvalidExif(f"Invalid CR3 structure: {e}")
    except Exception as e:
        logger.error(f"CR3: Unexpected error parsing: {e}")
        raise InvalidExif(f"Unexpected error during CR3 parsing: {e}")


def determine_type(fh: BinaryIO) -> Tuple[int, bytes, int]:
    # by default do not fake an EXIF beginning
    fake_exif = 0

    data = fh.read(12)
    if data[0:2] in [b"II", b"MM"]:
        # it's a TIFF file
        offset, endian = find_tiff_exif(fh)
    elif data[4:12] == b"ftypheic":
        fh.seek(0)
        heic = HEICExifFinder(fh)
        offset, endian = heic.find_exif()
        if offset == 0:
            offset, endian = find_heic_tiff(fh)
            # It's a HEIC file with a TIFF header
    elif data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        offset, endian = find_webp_exif(fh)
    elif data[0:2] == b"\xff\xd8":
        # it's a JPEG file
        offset, endian, fake_exif = find_jpeg_exif(fh, data, fake_exif)
    elif data[0:8] == b"\x89PNG\r\n\x1a\n":
        offset, endian = find_png_exif(fh, data)
    elif data[4:8] == b"ftyp" and data[8:12] == b"crx ":
        # It's a CR3 file
        logger.debug("CR3 format recognized")
        offset, endian = find_cr3_exif(fh)
    else:
        raise ExifNotFound("File format not recognized.")
    return offset, endian, fake_exif
