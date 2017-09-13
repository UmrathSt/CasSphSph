import tkinter as tk


my_window = tk.Tk()


r1 = tk.StringVar()
r2 = tk.StringVar()
d = tk.StringVar() # center_to_center_distance 

def retrieve_input():
    status_text = "worked"
    try:
        radius1 = float(r1.get())
    except:
        if not status_text == "worked":
            status_text.join("\ninvalid variable: Radius 1")
    try: 
        radius2 = float(r2.get())
    except:
        status_text = "invalid variable: Radius 2"    
    try:
        distance = float(d.get())
    except:
        status_text = "invalid variable: distance"
    status_label = tk.Label(my_window, text = status_text).grid()
    radii = [radius1, radius2]
    radii.sort()
    rS, rG = radii
    norm = 50
    r1_norm = rS/rG * norm
    r2_norm = norm
    d_norm = distance/rG * norm
    xoff = 10
    center_y = w.winfo_height()/2
    w.coords(sphereS, xoff, center_y+r1_norm,
                              2*r1_norm + xoff, center_y-r1_norm)
    w.itemconfig(sphereS, fill="grey")
    w.coords(sphereG, xoff + r1_norm + d_norm - r2_norm, center_y+r2_norm,
                xoff + r1_norm + d_norm + r2_norm, center_y-r2_norm)
    w.itemconfig(sphereG, fill="grey")
    return
    
my_window.geometry("800x600+200+200")
my_window.title("Casimir Interaction Calculator")
r1_label = tk.Label(my_window, text = "Radius 1").grid(row=0, column=0)
r1_entry = tk.Entry(my_window, textvariable = r1).grid(row=1, column=0)
r2_label = tk.Label(my_window, text = "Radius 2").grid(row=0, column=1)
r2_entry = tk.Entry(my_window, textvariable = r2).grid(row=1, column=1)
d_label = tk.Label(my_window, text = "Center Distance").grid(row=0, column=2)
d_entry = tk.Entry(my_window, textvariable = d).grid(row=1, column=2)

retrieve_button = tk.Button(my_window, text = "get variables", command = retrieve_input,
        fg = "black", bg = "grey").grid(row=2, column=0)


w = tk.Canvas(my_window, width=300, height=200)
w.grid(row=2, column=1)

sphereS = w.create_oval(10, w.winfo_height()/3,+60, w.winfo_height()*2/3, fill="grey")
sphereG = w.create_oval(10+100, w.winfo_height()/3,+50,  w.winfo_height()*2/3, fill="grey")



my_window.mainloop()

