from torchinfo import summary

def logoutput(model, model_conf,input_shape = None,depth=3,verbose=1,title="Model Summary"):
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)
    print("\n📐 Detailed Summary:")
    summary(
        model,
        input_size=input_shape,
        depth=depth,
        verbose=verbose,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )