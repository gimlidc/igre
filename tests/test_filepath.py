from stable.filepath import parse, change_suffix


def test_parse():
    filepath = "./myfolder/myfile.ext"
    folder, name, suffix = parse(filepath)
    assert folder == "./myfolder/"
    assert name == "myfile"
    assert suffix == "ext"


def test_change_suffix():
    filepath = "./myfolder/myfile.ext"
    newfile = change_suffix(filepath, "png")
    assert newfile == "./myfolder/myfile.png"
