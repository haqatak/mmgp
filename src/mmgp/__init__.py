import enum    
class profile_type(int, enum.Enum): 
    @staticmethod
    def tostr(v):
        if v == 1:
            s= "HighRAM_HighVRAM_Fastest" 
        elif v == 2:
            s ="HighRAM_LowVRAM_Fast"
        elif v == 3: 
            s = "LowRAM_HighVRAM_Medium"
        elif v == 4:
            s = "LowRAM_LowVRAM_Slow"
        else:
            s = "VerylowRAM_LowVRAM_Slowest"
        return s
    
    HighRAM_HighVRAM_Fastest = 1
    HighRAM_LowVRAM_Fast = 2
    LowRAM_HighVRAM_Medium = 3
    LowRAM_LowVRAM_Slow = 4
    VerylowRAM_LowVRAM_Slowest = 5

