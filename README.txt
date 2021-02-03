Autor: Kim Michael Ohnesorg, 22.01.2021
Bachelorarbeit 
"Laufzeitverkürzung eines Algorithmus für erweiterte Schärfentiefe
mit OpenCV CUDA für den Insektenscanner DISC3D" 
Optotechnik und Bildverabeitung, Hochschule Darmstadt
Betreuer: Prof. Dr. Stephan Neser

Mit dem beigefügten Code wird eine Zeitmessung durchgeführt.
Ein Schärfentiefeerweiterungs-Algorithmus wird auf der CPU und der GPU ausgeführt.
Die Messung kann einige Minuten in Anspruch nehmen, da der Code mehrfach ausgeführt wird.
Mit der unten aufgeführten CPU/GPU läuft die Zeitmessung ca. 15 Minuten. 
CPU Type: Intel(R) Core(TM) i7 CPU 860  @ 2.80GHz,Number of processors: 8
GPU: GeForce GTX 750 Ti
Die Messergebnisse werden als Text-Datei im Verzeichnis /Zeitmessung gespeichert.
Bitte senden Sie mir die Dateien zu, kimohnesorg@hotmail.de

Zum Ausführen der Zeitmessung ist die OpenCV mit CUDA, CMAKE, VisualStudio erforderlich,
zum Installieren von OpenCV CUDA folgen Sie zum Beispiel dem Link unten: 
https://medium.com/@mhfateen/build-opencv-4-4-0-with-cuda-gpu-support-on-windows-10-without-tears-aa85d470bcd0
In VisualStudio müssen einige Pfade gesetzt werden, folgen Sie dafür zum Beispiel dem Link unten:
https://aticleworld.com/install-opencv-with-visual-studio/

1. Installieren Sie OpenCV mit CUDA, CMAKE, VisualStudio 2019
2. Öffnen Sie die CMake GUI, fügen Sie den Pfad zum Verzeichnis /Zeitmessung unter "source code" und ein Build-Verzeichnis ein, klick "Configure", wähle als Generator Visual Studio 16 2019, klick "Finish", klicke "Configure" erneut und anschließend "Generate"
3. In VisualStudio wählen Sie Zeitmessung als StartUp Projekt
4. Setzen Sie die Pfade in VisualStudio, Achtung! Konfiguration: "Release" und Plattform: "x64", folge dafür den Anweisungen im Link
5. Starten Sie die Zeitmessung in VisualStudio durch klicken auf den grünen Pfeil "Lokaler Windows-Debugger"
6. Fügen Sie den Pfad zum Verzeichnis /Zeitmessung ein und bestätigen mit Enter, wenn die Anweisung in der Eingabeaufforderung erscheint
	Z.B. C:\Users\kimoh\Desktop\Zeitmessung
	Dann "C:\Users\kimoh\Desktop" einfügen