from fpdf import FPDF
from PIL import Image
import sys

pdf = FPDF()

dataset = sys.argv[1]

if(dataset == "svhn"):
    dir1 = "results/compare_attacks/svhn_conv_ae/"
    dir2 = "results/compare_attacks/svhn_ae_adv_trained/"
if(dataset == "mnist"):
    dir1 = "results/compare_attacks/mnist_ae/"
    dir2 = "results/compare_attacks/mnist_ae_adv_trained/"

cover = Image.open(dir1 + str(0) + ".png")
width, height = cover.size
N = 5

pdf = FPDF(unit = "pt", format = [width, height])
pdf.set_font('arial', 'B', 14.0)
pdf.add_page()
#pdf.cell (w = 0 , ln = 2, txt = "Comparitive Results after Adversarial Training")
#pdf.set_font('arial', 'B', 10.0)
pdf.cell (w = 0 , txt = "Left : for normal autoencoder ; Right : for autoencoder trained on "+str(N)+ " adversarial examples")
for i in range(0,15):
   
    pdf.image(dir1 + str(i) + ".png", x = 70, y = 50,  w = 370, h = 370)
    pdf.image(dir2 + str(i) + ".png", x = 450, y = 50, w = 370, h = 370)
    pdf.image(dir1 + "exp_" + str(i) + "graph1.png", x = 100, y = 380, w = 350, h = 280)
    pdf.image(dir2 + "exp_" + str(i) + "graph1.png", x = 470, y = 380, w = 350, h = 280)
    pdf.image(dir1 + "exp_" + str(i) + "graph2.png", x = 100, y = 680, w = 350, h = 280)
    pdf.image(dir2 + "exp_" + str(i) + "graph2.png", x = 470, y = 680, w = 350, h = 280)
    pdf.add_page()

x_coord = 70
y_coord = 30
for i in range(0,15):
    
    pdf.image(dir1 + str(i) + ".png", x = x_coord, y = y_coord,  w = 250, h = 250)
    pdf.image(dir2 + str(i) + ".png", x = x_coord + 220, y = y_coord, w = 250, h = 250)
    y_coord = y_coord + 220
if(dataset == "svhn"):
    pdfFileName = "compiled_results_svhn"
if(dataset == "mnist"):
    pdfFileName = "compiled_results_mnist"

dir = "results/compare_attacks/"
pdf.output(dir + pdfFileName + ".pdf", "F")


