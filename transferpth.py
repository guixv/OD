import torch
 
 
def replace_backbone_prefix(weight_key):
    weight_key = "backbone."+weight_key
    return weight_key
    
    # if weight_key.startswith("backbone."):
    #     return weight_key.replace("backbone.", "")
    # else:
    #     return weight_key
 
 
def save_modified_state_dict(filepath, new_state_dict):
    try:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            checkpoint = new_state_dict
            modified_filepath = filepath.replace('.pth', '_modified.pth')
            torch.save(checkpoint, modified_filepath)
            print("修改后的state_dict已成功保存到文件：", modified_filepath)
            return modified_filepath
        else:
            print("无法在文件中找到state_dict键，请确保你的.pth文件是PyTorch模型的权重文件。")
            return None
    except FileNotFoundError:
        print("文件路径无效，请检查输入的路径是否正确。")
        return None
    except Exception as e:
        print("保存修改后的state_dict时出现错误：", e)
        return None
 
 
def print_keys_from_pth(filepath):
    try:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = 'backbone.'+k # 去掉 `module.`
                new_state_dict[name] = v

            save_modified_state_dict(filepath, new_state_dict)
        else:
            print("无法在文件中找到state_dict键，请确保你的.pth文件是PyTorch模型的权重文件。")
    except FileNotFoundError:
        print("文件路径无效，请检查输入的路径是否正确。")
    except Exception as e:
        print("加载权重文件时出现错误：", e)
 
 
if __name__ == "__main__":
    filepath = "./MNV4.pth"
    print_keys_from_pth(filepath)
