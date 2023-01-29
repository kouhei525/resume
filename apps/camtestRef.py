import cv2


def main():
    #カメラの設定　デバイスIDは0
    cap = cv2.VideoCapture(0)

    #繰り返しのためのwhile文
    while True:
        #カメラからの画像取得
        ret, frame = cap.read()
        
        #カメラの画像の出力
        cv2.imshow('camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.imwrite("static/camera_capture.jpg", frame)
        #key1 =cv2.waitKey(1)& 0xFF
        elif key == ord('q'):
            break
            cap.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    #メモリを解放して終了するためのコマンド

if __name__ == "__main__":
    main()

