#################################################################################
# The tests below need to be rewritten once multimodal support is complete
# A pseudocode description has been written in comments to preserve notes about
# what was tested, for reference, in case we want to reproduct it in the new system
#################################################################################


def test_local_image():
    # 1. Create a mock model
    # 2. Add an image from the local filesystem to the model's prompt
    # 3. Validate that the model contains the image in its prompt
    pass


def test_local_image_not_found():
    # 1. Create a mock model
    # 2. Try to add a non-existing image from the local filesystem to the model's prompt
    # 3. Check for a file not found error, or other appropriate exception, to be thrown
    pass


def test_remote_image():
    # 1. Create a mock model
    # 2. Add a remote image from picsum using remote_image_url() utility function
    # 3. Validate that the model contains the image in its prompt
    pass


def test_remote_image_not_found():
    # 1. Create a mock model
    # 2. Try to add a non-existing remote image
    # 3. Catch an HTTPError or URLError from the model trying to fetch the image, which should result in a 404
    pass


def test_image_from_bytes():
    # 1. Create a mock model
    # 2. Download an image from remote_image_url() and save it as a binary file
    # 3. Read the binary file and add it to the model's prompt as an image
    # 3. Validate that the model contains the image in its prompt
    pass
