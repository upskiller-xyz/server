from ..processing.image_manager import ImgInput, ImgFname


def test_imginput_out():
    inp = [1, 2, 3]
    target = [4, 5, 6]
    img_input = ImgInput(inp=inp, target=target)
    assert img_input.out == (inp, target)


def test_imgfname_out():
    inp = "input.png"
    target = "target.png"
    img_fname = ImgFname(inp=inp, target=target)
    assert img_fname.out == (inp, target)
