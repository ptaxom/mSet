#include "view.h"




void init_window(int argc, char **argv)
{

    int width = 1200;
    int height = 768;

    GtkWidget *window;
    GtkWidget *darea;
    gtk_init(&argc, &argv);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    darea = gtk_drawing_area_new();

    gtk_container_add(GTK_CONTAINER(window), darea);

    gtk_window_set_title(GTK_WINDOW(window), "mSet renderer");

    gtk_window_set_resizable(GTK_WINDOW(window), FALSE);

    gtk_window_set_default_size(GTK_WINDOW(window), width, height);

    // gtk_widget_set_events(darea, gtk_widget_get_events(darea) | GDK_BUTTON_PRESS_MASK | GDK_SCROLL_MASK);

    // gtk_widget_add_events(darea, GDK_BUTTON_PRESS_MASK);
    // gtk_widget_add_events(window, GDK_SCROLL_MASK);

    g_signal_connect(G_OBJECT(window), "destroy", G_CALLBACK(gtk_main_quit), NULL);
    // g_signal_connect(G_OBJECT(darea), "draw", G_CALLBACK(draw), NULL);

    // g_signal_connect(darea, "button-press-event",
    //                  G_CALLBACK(button_press_event_cb), NULL);

    // g_signal_connect(window, "scroll-event", 
    //       G_CALLBACK(mouse_scroll), NULL);


    gtk_widget_show_all(window);
    gtk_main();
}