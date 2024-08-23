import argparse
from CrocoMarine_program_2024 import ObjectTracker
# python main.py --model_path "best(4).pt" --confidence_threshold 0.25 --classes 0 --save_output True --save_data True --origional_size False --mode 0 --input_path "seafloor_footage.mp4"
def main():
    parser = argparse.ArgumentParser(description="Run ObjectTracker with specified parameters.")
    parser.add_argument("-m" ,"--model_path"        , type=str  , required=True, help="Path to the model file.")
    parser.add_argument("--conf"                    , type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument("--classes"                 , type=int  , nargs='+', default=[0], help="List of classes to detect.")
    parser.add_argument("-svo" ,"--save_output"     , action='store_true',default=False  , help="Flag to save output.")
    parser.add_argument("-svd" ,"--save_data"       , action='store_true',default=False , help="Flag to save data.")
    parser.add_argument("-os" ,"--origional_size"   , action='store_true',default=False , help="Flag to use original size.")
    parser.add_argument("-sho", "--show_output"     , action='store_true',default=False  , help="show the model prediction output.")
    parser.add_argument("--mode"                    , type=int  , choices=[0, 1], default=0     , help="Mode of operation.")
    parser.add_argument("-ip","--input_path"        , type=str  , required=True , help="Path to the input video or image.")

    args = parser.parse_args()

    detector = ObjectTracker(
        model_path  =args.model_path,
        confidence_threshold=args.conf,
        classes     =args.classes,
        save_output =args.save_output,
        save_data   =args.save_data,
        origional_size=args.origional_size,
        mode        =args.mode,
        show_output =args.show_output
    )

    detector.run(args.input_path)

if __name__ == "__main__":
    main()